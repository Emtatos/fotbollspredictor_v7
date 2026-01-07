"""
Live-uppdatering av Form

Hämtar senaste matchresultat dagligen och uppdaterar features automatiskt.
"""
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
from data_loader import load_data_for_season
from feature_engineering import engineer_features

logger = logging.getLogger(__name__)


def update_data(leagues: list = ['E0', 'E1', 'E2', 'E3']) -> bool:
    """
    Uppdaterar data med senaste matchresultat
    
    Args:
        leagues: Ligor att uppdatera
        
    Returns:
        True om uppdatering lyckades
    """
    try:
        logger.info("Uppdaterar data för ligor: %s", leagues)
        
        # Hämta aktuell säsong
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # Fotbollssäsongen går från augusti till maj
        if current_month >= 8:
            season_code = f"{str(current_year)[2:]}{str(current_year + 1)[2:]}"
        else:
            season_code = f"{str(current_year - 1)[2:]}{str(current_year)[2:]}"
        
        logger.info("Aktuell säsong: %s", season_code)
        
        # Ladda data för alla ligor
        all_data = []
        for league in leagues:
            try:
                df = load_data_for_season(league, season_code)
                if df is not None and not df.empty:
                    all_data.append(df)
                    logger.info("Laddade %d matcher från %s", len(df), league)
            except Exception as e:
                logger.warning("Kunde inte ladda %s: %s", league, e)
        
        if not all_data:
            logger.error("Ingen data kunde laddas")
            return False
        
        # Kombinera all data
        df_combined = pd.concat(all_data, ignore_index=True)
        logger.info("Totalt %d matcher laddade", len(df_combined))
        
        # Spara rådata
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        raw_data_path = data_dir / f'raw_data_{season_code}.csv'
        df_combined.to_csv(raw_data_path, index=False)
        logger.info("Rådata sparad till: %s", raw_data_path)
        
        # Generera features
        logger.info("Genererar features...")
        df_features = engineer_features(df_combined)
        
        # Spara features
        features_path = data_dir / 'features.parquet'
        df_features.to_parquet(features_path, index=False)
        logger.info("Features sparade till: %s", features_path)
        
        return True
        
    except Exception as e:
        logger.error("Fel vid uppdatering: %s", e)
        return False


def get_last_update_time() -> datetime:
    """
    Hämtar tidpunkt för senaste uppdatering
    
    Returns:
        Datetime för senaste uppdatering
    """
    update_file = Path('data/.last_update')
    
    if update_file.exists():
        try:
            timestamp = float(update_file.read_text().strip())
            return datetime.fromtimestamp(timestamp)
        except:
            pass
    
    return datetime(1970, 1, 1)


def set_last_update_time():
    """
    Sparar tidpunkt för senaste uppdatering
    """
    update_file = Path('data/.last_update')
    update_file.parent.mkdir(exist_ok=True)
    update_file.write_text(str(datetime.now().timestamp()))


def should_update(hours: int = 24) -> bool:
    """
    Kontrollerar om data bör uppdateras
    
    Args:
        hours: Antal timmar sedan senaste uppdatering
        
    Returns:
        True om uppdatering behövs
    """
    last_update = get_last_update_time()
    time_since_update = datetime.now() - last_update
    
    return time_since_update.total_seconds() > (hours * 3600)


def auto_update_if_needed(hours: int = 24) -> bool:
    """
    Uppdaterar automatiskt om det behövs
    
    Args:
        hours: Antal timmar mellan uppdateringar
        
    Returns:
        True om uppdatering gjordes
    """
    if should_update(hours):
        logger.info("Data är gammal, uppdaterar...")
        success = update_data()
        
        if success:
            set_last_update_time()
            logger.info("Uppdatering klar!")
            return True
        else:
            logger.error("Uppdatering misslyckades")
            return False
    else:
        last_update = get_last_update_time()
        logger.info("Data är aktuell (senast uppdaterad: %s)", last_update)
        return False


# Exempel på användning
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=== LIVE-UPPDATERING AV DATA ===\n")
    
    # Kontrollera om uppdatering behövs
    last_update = get_last_update_time()
    print(f"Senaste uppdatering: {last_update}")
    
    time_since = datetime.now() - last_update
    hours_since = time_since.total_seconds() / 3600
    print(f"Tid sedan uppdatering: {hours_since:.1f} timmar")
    print()
    
    if should_update(24):
        print("✅ Uppdatering behövs (>24 timmar sedan senast)")
        print("Uppdaterar data...")
        print()
        
        success = update_data()
        
        if success:
            set_last_update_time()
            print("\n✅ Uppdatering klar!")
        else:
            print("\n❌ Uppdatering misslyckades")
    else:
        print("ℹ️  Data är aktuell, ingen uppdatering behövs")
    
    print("\n💡 TIPS:")
    print("- Kör detta script dagligen för att hålla data uppdaterad")
    print("- Använd cron job för automatisk uppdatering:")
    print("  0 6 * * * cd /path/to/project && python live_update.py")
    print("- Detta uppdaterar data kl 06:00 varje dag")
