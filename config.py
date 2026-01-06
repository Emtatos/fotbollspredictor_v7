"""
Centraliserad konfiguration för fotbollspredictor_v7

Denna modul hanterar all konfiguration och miljövariabler på ett säkert sätt.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Ladda miljövariabler från .env
load_dotenv()


class Config:
    """Konfigurationsklass för applikationen"""
    
    # Projektstruktur
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    
    # Datakällor
    FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"
    API_FOOTBALL_URL = "https://v3.football.api-sports.io"
    
    # Ligor
    LEAGUES = ["E0", "E1", "E2"]  # Premier League, Championship, League One
    
    # Modellparametrar
    MODEL_PARAMS = {
        "objective": "multi:softprob",
        "n_estimators": 250,
        "learning_rate": 0.1,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss",
        "n_jobs": -1,
        "random_state": 42,
    }
    
    # ELO-parametrar
    ELO_K_FACTOR = 20
    ELO_INITIAL_RATING = 1500
    
    # Form-parametrar
    FORM_WINDOW = 5  # Antal matcher för form-beräkning
    
    # API-nycklar (säkert hämtade)
    @staticmethod
    def get_api_football_key() -> Optional[str]:
        """Hämtar API-nyckel för api-football.com"""
        return os.getenv("API_FOOTBALL_KEY")
    
    @staticmethod
    def get_openai_key() -> Optional[str]:
        """Hämtar API-nyckel för OpenAI"""
        return os.getenv("OPENAI_API_KEY")
    
    @staticmethod
    def has_api_football_key() -> bool:
        """Kontrollerar om API-nyckel för api-football finns"""
        return bool(Config.get_api_football_key())
    
    @staticmethod
    def has_openai_key() -> bool:
        """Kontrollerar om OpenAI API-nyckel finns"""
        return bool(Config.get_openai_key())
    
    # Loggning
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Streamlit
    STREAMLIT_PAGE_TITLE = "Fotbollspredictor v7"
    STREAMLIT_PAGE_ICON = "⚽"
    STREAMLIT_LAYOUT = "wide"
    
    @classmethod
    def ensure_directories(cls):
        """Skapar nödvändiga mappar om de inte finns"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Skapa mappar vid import
Config.ensure_directories()
