"""
Tester för kalibrering av sannolikheter (Step 2A)

Verifierar att:
- Kalibrerad modell kan laddas
- predict_proba returnerar korrekt shape (n, 3)
- Varje rad summerar till ~1.0
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from model_handler import train_and_save_model, load_model, MODEL_BASE_FILENAME, MODEL_CALIBRATED_FILENAME
from feature_engineering import create_features
from schema import FEATURE_COLUMNS, encode_league
from sklearn.calibration import CalibratedClassifierCV


class TestCalibration:
    """Tester för kalibrerad modell"""
    
    @pytest.fixture
    def training_data(self):
        """Skapa träningsdata med alla tre resultattyper"""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        teams = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Tottenham", "Man Utd"]
        
        data = []
        for i in range(60):
            home = teams[i % len(teams)]
            away = teams[(i + 2) % len(teams)]
            
            home_strength = teams.index(home)
            away_strength = teams.index(away)
            
            if home_strength < away_strength:
                ftr = "H" if np.random.random() > 0.3 else "D"
                fthg, ftag = 2, 1
            elif home_strength > away_strength:
                ftr = "A" if np.random.random() > 0.3 else "D"
                fthg, ftag = 1, 2
            else:
                ftr = "D"
                fthg, ftag = 1, 1
            
            data.append({
                "Date": dates[i],
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": fthg,
                "FTAG": ftag,
                "FTR": ftr,
                "League": "E0"
            })
        
        df = pd.DataFrame(data)
        return create_features(df)
    
    def test_calibrated_model_is_saved(self, training_data):
        """Testar att både bas- och kalibrerad modell sparas"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.joblib"
            
            model = train_and_save_model(training_data, model_path)
            
            assert model is not None
            
            base_path = Path(temp_dir) / MODEL_BASE_FILENAME
            calibrated_path = Path(temp_dir) / MODEL_CALIBRATED_FILENAME
            
            assert base_path.exists(), f"Basmodell saknas: {base_path}"
            assert calibrated_path.exists(), f"Kalibrerad modell saknas: {calibrated_path}"
    
    def test_calibrated_model_can_be_loaded(self, training_data):
        """Testar att kalibrerad modell kan laddas"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.joblib"
            
            train_and_save_model(training_data, model_path)
            
            loaded_model = load_model(model_path)
            
            assert loaded_model is not None
            assert isinstance(loaded_model, CalibratedClassifierCV), \
                f"Förväntade CalibratedClassifierCV, fick {type(loaded_model)}"
    
    def test_predict_proba_shape(self, training_data):
        """Testar att predict_proba returnerar shape (n, 3)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.joblib"
            
            model = train_and_save_model(training_data, model_path)
            
            df_pred = training_data.copy()
            df_pred["League"] = df_pred["League"].apply(encode_league)
            X_test = df_pred[FEATURE_COLUMNS].iloc[:5].values
            
            probs = model.predict_proba(X_test)
            
            assert probs.shape == (5, 3), f"Förväntade shape (5, 3), fick {probs.shape}"
    
    def test_predict_proba_sums_to_one(self, training_data):
        """Testar att varje rad i predict_proba summerar till ~1.0"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.joblib"
            
            model = train_and_save_model(training_data, model_path)
            
            df_pred = training_data.copy()
            df_pred["League"] = df_pred["League"].apply(encode_league)
            X_test = df_pred[FEATURE_COLUMNS].iloc[:10].values
            
            probs = model.predict_proba(X_test)
            
            for i, row in enumerate(probs):
                row_sum = row.sum()
                assert abs(row_sum - 1.0) < 1e-6, \
                    f"Rad {i} summerar till {row_sum}, förväntade ~1.0"
    
    def test_predict_proba_valid_probabilities(self, training_data):
        """Testar att alla sannolikheter är mellan 0 och 1"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.joblib"
            
            model = train_and_save_model(training_data, model_path)
            
            df_pred = training_data.copy()
            df_pred["League"] = df_pred["League"].apply(encode_league)
            X_test = df_pred[FEATURE_COLUMNS].iloc[:10].values
            
            probs = model.predict_proba(X_test)
            
            assert np.all(probs >= 0), "Negativa sannolikheter hittades"
            assert np.all(probs <= 1), "Sannolikheter > 1 hittades"
    
    def test_loaded_model_predict_proba(self, training_data):
        """Testar att laddad kalibrerad modell ger korrekt predict_proba"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.joblib"
            
            train_and_save_model(training_data, model_path)
            
            loaded_model = load_model(model_path)
            
            df_pred = training_data.copy()
            df_pred["League"] = df_pred["League"].apply(encode_league)
            X_test = df_pred[FEATURE_COLUMNS].iloc[:5].values
            
            probs = loaded_model.predict_proba(X_test)
            
            assert probs.shape == (5, 3), f"Förväntade shape (5, 3), fick {probs.shape}"
            
            for i, row in enumerate(probs):
                row_sum = row.sum()
                assert abs(row_sum - 1.0) < 1e-6, \
                    f"Rad {i} summerar till {row_sum}, förväntade ~1.0"
    
    def test_calibration_method_sigmoid(self, training_data):
        """Testar att sigmoid-kalibrering fungerar (default)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.joblib"
            
            model = train_and_save_model(training_data, model_path, calibration_method="sigmoid")
            
            assert model is not None
            assert isinstance(model, CalibratedClassifierCV)
    
    def test_calibration_method_isotonic(self, training_data):
        """Testar att isotonic-kalibrering fungerar"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.joblib"
            
            model = train_and_save_model(training_data, model_path, calibration_method="isotonic")
            
            assert model is not None
            assert isinstance(model, CalibratedClassifierCV)
