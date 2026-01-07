import pandas as pd
from collections import defaultdict, deque
import numpy as np
import logging

# Importera vår hjälpfunktion
from utils import normalize_team_name

# Få tag i logger-instansen
logger = logging.getLogger(__name__)

def _calculate_form(df: pd.DataFrame) -> pd.DataFrame:
    # (Denna funktion är oförändrad)
    team_points = defaultdict(lambda: deque(maxlen=5))
    team_gd = defaultdict(lambda: deque(maxlen=5))
    df['HomeFormPts'] = np.nan
    df['HomeFormGD'] = np.nan
    df['AwayFormPts'] = np.nan
    df['AwayFormGD'] = np.nan
    for index, row in df.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        if len(team_points[home_team]) > 0:
            df.loc[index, 'HomeFormPts'] = np.mean(list(team_points[home_team]))
            df.loc[index, 'HomeFormGD'] = np.mean(list(team_gd[home_team]))
        if len(team_points[away_team]) > 0:
            df.loc[index, 'AwayFormPts'] = np.mean(list(team_points[away_team]))
            df.loc[index, 'AwayFormGD'] = np.mean(list(team_gd[away_team]))
        home_gd = row['FTHG'] - row['FTAG']
        if row['FTR'] == 'H': home_pts, away_pts = 3, 0
        elif row['FTR'] == 'D': home_pts, away_pts = 1, 1
        else: home_pts, away_pts = 0, 3
        team_points[home_team].append(home_pts)
        team_points[away_team].append(away_pts)
        team_gd[home_team].append(home_gd)
        team_gd[away_team].append(-home_gd)
    df.fillna(0, inplace=True)
    return df

def _calculate_elo(df: pd.DataFrame, k_factor: int = 20) -> pd.DataFrame:
    # (Denna funktion är oförändrad)
    elo_ratings = defaultdict(lambda: 1500)
    df['HomeElo'], df['AwayElo'] = np.nan, np.nan
    for index, row in df.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        home_elo_before, away_elo_before = elo_ratings[home_team], elo_ratings[away_team]
        df.loc[index, 'HomeElo'], df.loc[index, 'AwayElo'] = home_elo_before, away_elo_before
        expected_home = 1 / (1 + 10 ** ((away_elo_before - home_elo_before) / 400))
        if row['FTR'] == 'H': actual_home = 1.0
        elif row['FTR'] == 'D': actual_home = 0.5
        else: actual_home = 0.0
        new_home_elo = home_elo_before + k_factor * (actual_home - expected_home)
        new_away_elo = away_elo_before + k_factor * ((1 - actual_home) - (1 - expected_home))
        elo_ratings[home_team], elo_ratings[away_team] = new_home_elo, new_away_elo
    df.fillna(1500, inplace=True)
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df_copy = df.copy()
    df_copy.sort_values(by="Date", inplace=True, ascending=True)
    df_copy.reset_index(drop=True, inplace=True)

    # ======================================================================
    #  START PÅ FELSÖKNINGSLOGIK
    # ======================================================================
    
    logger.info("--- STARTAR FELSÖKNING AV NORMALISERING ---")
    
    # Logga unika lagnamn FÖRE normalisering
    try:
        raw_unique_teams = pd.unique(df_copy[['HomeTeam', 'AwayTeam']].values.ravel('K'))
        logger.info(f"Unika lagnamn i rådatan (FÖRE normalisering): {sorted(raw_unique_teams)}")
    except Exception as e:
        logger.error(f"Kunde inte logga råa lagnamn: {e}")

    # Skapa en wrapper-funktion för att logga varje ändring
    def log_and_normalize(name):
        normalized = normalize_team_name(name)
        # Logga endast om en ändring faktiskt sker
        if name != normalized:
            logger.info(f"NORMALIZATION TRACE: '{name}' -> '{normalized}'")
        return normalized

    # Normalisera lagnamn med den nya loggningsfunktionen
    df_copy['HomeTeam'] = df_copy['HomeTeam'].apply(log_and_normalize)
    df_copy['AwayTeam'] = df_copy['AwayTeam'].apply(log_and_normalize)

    # Logga unika lagnamn EFTER normalisering
    try:
        processed_unique_teams = pd.unique(df_copy[['HomeTeam', 'AwayTeam']].values.ravel('K'))
        logger.info(f"Unika lagnamn i datan (EFTER normalisering): {sorted(processed_unique_teams)}")
    except Exception as e:
        logger.error(f"Kunde inte logga bearbetade lagnamn: {e}")
        
    logger.info("--- AVSLUTAR FELSÖKNING AV NORMALISERING ---")
    
    # ======================================================================
    #  SLUT PÅ FELSÖKNINGSLOGIK
    # ======================================================================

    # Beräkna features (som vanligt)
    df_with_form = _calculate_form(df_copy)
    df_with_elo = _calculate_elo(df_with_form)
    
    added_cols = [c for c in df_with_elo.columns if c not in df.columns]
    logger.info("Feature engineering slutförd. Nya kolumner: %s", added_cols)
    
    return df_with_elo
