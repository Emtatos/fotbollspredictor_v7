import pandas as pd
from collections import defaultdict, deque
import numpy as np
import logging

# Importera vår hjälpfunktion
from utils import normalize_team_name

# Få tag i logger-instansen
logger = logging.getLogger(__name__)

def _calculate_form(df: pd.DataFrame) -> pd.DataFrame:
    """Beräknar form baserat på senaste 5 matcherna (poäng och målskillnad)"""
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
        if row['FTR'] == 'H': 
            home_pts, away_pts = 3, 0
        elif row['FTR'] == 'D': 
            home_pts, away_pts = 1, 1
        else: 
            home_pts, away_pts = 0, 3
        
        team_points[home_team].append(home_pts)
        team_points[away_team].append(away_pts)
        team_gd[home_team].append(home_gd)
        team_gd[away_team].append(-home_gd)
    
    df.fillna(0, inplace=True)
    return df


def _calculate_home_away_form(df: pd.DataFrame) -> pd.DataFrame:
    """Beräknar separat form för hemma- och bortamatcher"""
    home_points = defaultdict(lambda: deque(maxlen=5))
    away_points = defaultdict(lambda: deque(maxlen=5))
    
    df['HomeFormHome'] = np.nan  # Form i hemmamatcher
    df['AwayFormAway'] = np.nan  # Form i bortamatcher
    
    for index, row in df.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        
        # Sätt form baserat på tidigare hemma/bortamatcher
        if len(home_points[home_team]) > 0:
            df.loc[index, 'HomeFormHome'] = np.mean(list(home_points[home_team]))
        if len(away_points[away_team]) > 0:
            df.loc[index, 'AwayFormAway'] = np.mean(list(away_points[away_team]))
        
        # Uppdatera efter matchen
        if row['FTR'] == 'H':
            home_points[home_team].append(3)
            away_points[away_team].append(0)
        elif row['FTR'] == 'D':
            home_points[home_team].append(1)
            away_points[away_team].append(1)
        else:
            home_points[home_team].append(0)
            away_points[away_team].append(3)
    
    df.fillna(0, inplace=True)
    return df


def _calculate_goal_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Beräknar målstatistik (gjorda och insläppta mål senaste 5 matcherna)"""
    team_goals_for = defaultdict(lambda: deque(maxlen=5))
    team_goals_against = defaultdict(lambda: deque(maxlen=5))
    
    df['HomeGoalsFor'] = np.nan
    df['HomeGoalsAgainst'] = np.nan
    df['AwayGoalsFor'] = np.nan
    df['AwayGoalsAgainst'] = np.nan
    
    for index, row in df.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        
        # Sätt statistik före matchen
        if len(team_goals_for[home_team]) > 0:
            df.loc[index, 'HomeGoalsFor'] = np.mean(list(team_goals_for[home_team]))
            df.loc[index, 'HomeGoalsAgainst'] = np.mean(list(team_goals_against[home_team]))
        if len(team_goals_for[away_team]) > 0:
            df.loc[index, 'AwayGoalsFor'] = np.mean(list(team_goals_for[away_team]))
            df.loc[index, 'AwayGoalsAgainst'] = np.mean(list(team_goals_against[away_team]))
        
        # Uppdatera efter matchen
        team_goals_for[home_team].append(row['FTHG'])
        team_goals_against[home_team].append(row['FTAG'])
        team_goals_for[away_team].append(row['FTAG'])
        team_goals_against[away_team].append(row['FTHG'])
    
    df.fillna(0, inplace=True)
    return df


def _calculate_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """Beräknar vinst/förlust-sviter"""
    team_streak = defaultdict(int)  # Positiv = vinster, negativ = förluster, 0 = oavgjort
    
    df['HomeStreak'] = 0
    df['AwayStreak'] = 0
    
    for index, row in df.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        
        # Sätt streak före matchen
        df.loc[index, 'HomeStreak'] = team_streak[home_team]
        df.loc[index, 'AwayStreak'] = team_streak[away_team]
        
        # Uppdatera streak efter matchen
        if row['FTR'] == 'H':
            # Hemmavinst
            if team_streak[home_team] >= 0:
                team_streak[home_team] += 1
            else:
                team_streak[home_team] = 1
            
            # Bortaförlust
            if team_streak[away_team] <= 0:
                team_streak[away_team] -= 1
            else:
                team_streak[away_team] = -1
        
        elif row['FTR'] == 'A':
            # Bortavinst
            if team_streak[away_team] >= 0:
                team_streak[away_team] += 1
            else:
                team_streak[away_team] = 1
            
            # Hemmaförlust
            if team_streak[home_team] <= 0:
                team_streak[home_team] -= 1
            else:
                team_streak[home_team] = -1
        
        else:  # Oavgjort
            team_streak[home_team] = 0
            team_streak[away_team] = 0
    
    return df


def _calculate_elo(df: pd.DataFrame, k_factor: int = 20) -> pd.DataFrame:
    """Beräknar ELO-rating för lag"""
    elo_ratings = defaultdict(lambda: 1500)
    df['HomeElo'], df['AwayElo'] = np.nan, np.nan
    
    for index, row in df.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        home_elo_before, away_elo_before = elo_ratings[home_team], elo_ratings[away_team]
        df.loc[index, 'HomeElo'], df.loc[index, 'AwayElo'] = home_elo_before, away_elo_before
        
        expected_home = 1 / (1 + 10 ** ((away_elo_before - home_elo_before) / 400))
        
        if row['FTR'] == 'H': 
            actual_home = 1.0
        elif row['FTR'] == 'D': 
            actual_home = 0.5
        else: 
            actual_home = 0.0
        
        new_home_elo = home_elo_before + k_factor * (actual_home - expected_home)
        new_away_elo = away_elo_before + k_factor * ((1 - actual_home) - (1 - expected_home))
        elo_ratings[home_team], elo_ratings[away_team] = new_home_elo, new_away_elo
    
    df.fillna(1500, inplace=True)
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Huvudfunktion som skapar alla features"""
    if df.empty:
        return pd.DataFrame()

    df_copy = df.copy()
    df_copy.sort_values(by="Date", inplace=True, ascending=True)
    df_copy.reset_index(drop=True, inplace=True)

    # Normalisera lagnamn
    logger.info("--- STARTAR FEATURE ENGINEERING V2 ---")
    
    def log_and_normalize(name):
        normalized = normalize_team_name(name)
        if name != normalized:
            logger.info(f"NORMALIZATION: '{name}' -> '{normalized}'")
        return normalized

    df_copy['HomeTeam'] = df_copy['HomeTeam'].apply(log_and_normalize)
    df_copy['AwayTeam'] = df_copy['AwayTeam'].apply(log_and_normalize)

    # Beräkna alla features
    logger.info("Beräknar grundläggande form...")
    df_with_form = _calculate_form(df_copy)
    
    logger.info("Beräknar hemma/borta-specifik form...")
    df_with_home_away = _calculate_home_away_form(df_with_form)
    
    logger.info("Beräknar målstatistik...")
    df_with_goals = _calculate_goal_stats(df_with_home_away)
    
    logger.info("Beräknar streaks...")
    df_with_streaks = _calculate_streaks(df_with_goals)
    
    logger.info("Beräknar ELO-rating...")
    df_final = _calculate_elo(df_with_streaks)
    
    added_cols = [c for c in df_final.columns if c not in df.columns]
    logger.info(f"Feature engineering slutförd. Nya kolumner ({len(added_cols)}): {added_cols}")
    
    return df_final
