"""
Quick-fix script för att träna om modellen med 27 features (inklusive skade-features).
Detta script kan köras direkt från Streamlit-appen.
"""

import logging
from pathlib import Path
import pandas as pd
from feature_engineering import create_features
from model_handler import train_and_save_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retrain_with_injury_features():
    """
    Tränar om modellen med alla 27 features (21 original + 6 skade-features).
    Använder befintlig features.parquet och lägger till skade-features.
    """
    logger.info("=== Startar snabb omträning med 27 features ===")
    
    # 1. Ladda befintlig features
    features_path = Path("data/features.parquet")
    if not features_path.exists():
        raise FileNotFoundError("features.parquet saknas! Kör main.py först.")
    
    logger.info("Laddar befintlig features.parquet...")
    df = pd.read_parquet(features_path)
    logger.info(f"Laddade {len(df)} rader")
    
    # 2. Lägg till skade-features om de saknas
    injury_features = ['InjuredPlayers_Home', 'InjuredPlayers_Away', 
                      'KeyPlayersOut_Home', 'KeyPlayersOut_Away',
                      'InjurySeverity_Home', 'InjurySeverity_Away']
    
    missing_features = [f for f in injury_features if f not in df.columns]
    
    if missing_features:
        logger.info(f"Lägger till saknade skade-features: {missing_features}")
        for feature in missing_features:
            df[feature] = 0  # Default värde när ingen skadedata finns
        logger.info("Skade-features tillagda (default=0)")
    else:
        logger.info("Alla skade-features finns redan!")
    
    # 3. Spara uppdaterad features
    logger.info("Sparar uppdaterad features.parquet...")
    df.to_parquet(features_path, index=False)
    
    # 4. Träna ny modell
    from datetime import datetime
    now = datetime.now()
    year = now.year
    if now.month < 7:
        season_code = f"{str(year-1)[-2:]}{str(year)[-2:]}"
    else:
        season_code = f"{str(year)[-2:]}{str(year+1)[-2:]}"
    
    model_filename = f"xgboost_model_v7_{season_code}.joblib"
    model_path = Path("models") / model_filename
    
    logger.info(f"Tränar ny modell med {len(df.columns)} features...")
    logger.info(f"Features: {list(df.columns)}")
    
    model = train_and_save_model(df, model_path)
    
    logger.info(f"✅ Modell tränad och sparad: {model_path}")
    logger.info(f"✅ Modellen har nu {model.n_features_in_} features")
    
    return model_path

if __name__ == "__main__":
    retrain_with_injury_features()
