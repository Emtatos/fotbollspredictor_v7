# main.py
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd  # valfritt men praktiskt vid ev. debugging/inspektion

from data_loader import download_season_data
from data_processing import normalize_csv_data
from feature_engineering import create_features
from model_handler import train_and_save_model, load_model

# Konfigurera en global logger för huvudskriptet
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_current_season_code() -> str:
    """Returnerar säsongskoden för den pågående säsongen (t.ex. '2425')."""
    now = datetime.now()
    year = now.year
    # Före juli → säsongen som slutar i år (startade förra året)
    if now.month < 7:
        start_year = year - 1
        end_year = year
    else:
        start_year = year
        end_year = year + 1
    return f"{str(start_year)[-2:]}{str(end_year)[-2:]}"

def run_pipeline():
    """
    Kör hela databehandlings- och modellträningspipelinen.
    """
    logger.info("=============================================")
    logger.info("=== Startar pipeline för fotbollsmodell ===")
    logger.info("=============================================")

    # --- 1. Konfiguration ---
    SEASON = get_current_season_code()
    LEAGUES = ["E0", "E1", "E2"]  # Premier League, Championship, League One
    MODEL_FILENAME = f"xgboost_model_v7_{SEASON}.joblib"
    model_path = Path("models") / MODEL_FILENAME

    logger.info("Säsong: %s, Ligor: %s", SEASON, LEAGUES)
    logger.info("Modellfil: %s", model_path)

    # --- 2. Hämta rådata ---
    logger.info("--- Steg 1: Hämtar rådata ---")
    downloaded_files = download_season_data(season_code=SEASON, leagues=LEAGUES)
    if not downloaded_files:
        logger.error("Inga filer kunde laddas ner. Avbryter pipeline.")
        return
    logger.info("Hämtade %d filer.", len(downloaded_files))

    # --- 3. Bearbeta och normalisera data ---
    logger.info("--- Steg 2: Bearbetar och normaliserar CSV-data ---")
    df_clean = normalize_csv_data(file_paths=downloaded_files)
    if df_clean.empty:
        logger.error("Ingen data kunde normaliseras. Avbryter pipeline.")
        return
    logger.info("Normaliserade %d matcher.", len(df_clean))

    # --- 4. Skapa features ---
    logger.info("--- Steg 3: Skapar features (form och ELO) ---")
    df_features = create_features(df=df_clean)
    if df_features.empty:
        logger.error("Kunde inte skapa features. Avbryter pipeline.")
        return
    logger.info("Skapade features. DataFrame har nu %d kolumner.", len(df_features.columns))

    # --- 5. Träna och spara modell ---
    logger.info("--- Steg 4: Tränar och sparar modell ---")
    trained_model = train_and_save_model(df_features=df_features, model_path=model_path)
    if trained_model is None:
        logger.error("Modellträningen misslyckades. Avbryter pipeline.")
        return
    logger.info("Modellen har tränats och sparats framgångsrikt.")
    
    # --- 6. Verifiera laddning av modell (valfritt test) ---
    logger.info("--- Steg 5: Verifierar att modellen kan laddas ---")
    loaded_model = load_model(model_path=model_path)
    if loaded_model:
        logger.info("Verifiering lyckades! Modellen kan laddas från disk.")
    else:
        logger.warning("Verifiering misslyckades. Något är fel med den sparade modellfilen.")

    logger.info("==========================================")
    logger.info("=== Pipelinen har körts färdigt! ===")
    logger.info("==========================================")

if __name__ == "__main__":
    run_pipeline()
