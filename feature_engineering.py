import pandas as pd
from collections import defaultdict, deque
import numpy as np
import logging

# Importera vår hjälpfunktion
from utils import normalize_team_name

# Få tag i logger-instansen
logger = logging.getLogger(__name__)

# Import för skade-features (optional)
try:
    from injury_scraper import InjuryDataFetcher, get_injury_features_for_match
    INJURY_FEATURES_AVAILABLE = True
except ImportError:
    INJURY_FEATURES_AVAILABLE = False
    logger.warning("injury_scraper inte tillgänglig - skade-features inaktiverade")

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
    
    df['HomeFormHome'] = np.nan
    df['AwayFormAway'] = np.nan
    
    for index, row in df.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        
        if len(home_points[home_team]) > 0:
            df.loc[index, 'HomeFormHome'] = np.mean(list(home_points[home_team]))
        if len(away_points[away_team]) > 0:
            df.loc[index, 'AwayFormAway'] = np.mean(list(away_points[away_team]))
        
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
        
        if len(team_goals_for[home_team]) > 0:
            df.loc[index, 'HomeGoalsFor'] = np.mean(list(team_goals_for[home_team]))
            df.loc[index, 'HomeGoalsAgainst'] = np.mean(list(team_goals_against[home_team]))
        if len(team_goals_for[away_team]) > 0:
            df.loc[index, 'AwayGoalsFor'] = np.mean(list(team_goals_for[away_team]))
            df.loc[index, 'AwayGoalsAgainst'] = np.mean(list(team_goals_against[away_team]))
        
        team_goals_for[home_team].append(row['FTHG'])
        team_goals_against[home_team].append(row['FTAG'])
        team_goals_for[away_team].append(row['FTAG'])
        team_goals_against[away_team].append(row['FTHG'])
    
    df.fillna(0, inplace=True)
    return df


def _calculate_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """Beräknar vinst/förlust-sviter"""
    team_streak = defaultdict(int)
    
    df['HomeStreak'] = 0
    df['AwayStreak'] = 0
    
    for index, row in df.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        
        df.loc[index, 'HomeStreak'] = team_streak[home_team]
        df.loc[index, 'AwayStreak'] = team_streak[away_team]
        
        if row['FTR'] == 'H':
            if team_streak[home_team] >= 0:
                team_streak[home_team] += 1
            else:
                team_streak[home_team] = 1
            
            if team_streak[away_team] <= 0:
                team_streak[away_team] -= 1
            else:
                team_streak[away_team] = -1
        
        elif row['FTR'] == 'A':
            if team_streak[away_team] >= 0:
                team_streak[away_team] += 1
            else:
                team_streak[away_team] = 1
            
            if team_streak[home_team] <= 0:
                team_streak[home_team] -= 1
            else:
                team_streak[home_team] = -1
        
        else:
            team_streak[home_team] = 0
            team_streak[away_team] = 0
    
    return df


def _calculate_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    """Beräknar Head-to-Head statistik (senaste 5 mötena mellan lagen)"""
    h2h_history = defaultdict(lambda: deque(maxlen=5))
    
    df['H2H_HomeWins'] = 0
    df['H2H_Draws'] = 0
    df['H2H_AwayWins'] = 0
    df['H2H_HomeGoalDiff'] = 0.0
    
    for index, row in df.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        matchup = tuple(sorted([home_team, away_team]))
        
        # Beräkna statistik från tidigare möten
        if len(h2h_history[matchup]) > 0:
            home_wins = 0
            away_wins = 0
            draws = 0
            goal_diff = 0
            
            for prev_match in h2h_history[matchup]:
                prev_home, prev_away, prev_result, prev_gd = prev_match
                
                # Justera för om lagen bytt plats
                if prev_home == home_team:
                    if prev_result == 'H':
                        home_wins += 1
                    elif prev_result == 'A':
                        away_wins += 1
                    else:
                        draws += 1
                    goal_diff += prev_gd
                else:  # Lagen har bytt plats
                    if prev_result == 'H':
                        away_wins += 1
                    elif prev_result == 'A':
                        home_wins += 1
                    else:
                        draws += 1
                    goal_diff -= prev_gd
            
            df.loc[index, 'H2H_HomeWins'] = home_wins
            df.loc[index, 'H2H_Draws'] = draws
            df.loc[index, 'H2H_AwayWins'] = away_wins
            df.loc[index, 'H2H_HomeGoalDiff'] = goal_diff / len(h2h_history[matchup])
        
        # Spara denna match i historiken
        h2h_history[matchup].append((home_team, away_team, row['FTR'], row['FTHG'] - row['FTAG']))
    
    return df


def _calculate_league_position(df: pd.DataFrame) -> pd.DataFrame:
    """Beräknar ligaposition baserat på poäng och målskillnad"""
    # Gruppera per säsong och liga (om det finns)
    # För enkelhetens skull: beräkna position baserat på alla matcher hittills
    
    team_stats = defaultdict(lambda: {'points': 0, 'gd': 0, 'games': 0})
    
    df['HomePosition'] = 0
    df['AwayPosition'] = 0
    df['PositionDiff'] = 0
    
    for index, row in df.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        
        # Beräkna nuvarande positioner
        if team_stats[home_team]['games'] > 0 and team_stats[away_team]['games'] > 0:
            # Sortera alla lag efter poäng och målskillnad
            sorted_teams = sorted(
                team_stats.items(),
                key=lambda x: (x[1]['points'], x[1]['gd']),
                reverse=True
            )
            
            # Hitta positioner
            home_pos = next((i+1 for i, (team, _) in enumerate(sorted_teams) if team == home_team), 0)
            away_pos = next((i+1 for i, (team, _) in enumerate(sorted_teams) if team == away_team), 0)
            
            df.loc[index, 'HomePosition'] = home_pos
            df.loc[index, 'AwayPosition'] = away_pos
            df.loc[index, 'PositionDiff'] = away_pos - home_pos  # Positivt = hemmalaget högre placerat
        
        # Uppdatera statistik efter matchen
        home_gd = row['FTHG'] - row['FTAG']
        if row['FTR'] == 'H':
            team_stats[home_team]['points'] += 3
        elif row['FTR'] == 'D':
            team_stats[home_team]['points'] += 1
            team_stats[away_team]['points'] += 1
        else:
            team_stats[away_team]['points'] += 3
        
        team_stats[home_team]['gd'] += home_gd
        team_stats[away_team]['gd'] -= home_gd
        team_stats[home_team]['games'] += 1
        team_stats[away_team]['games'] += 1
    
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
    logger.info("--- STARTAR FEATURE ENGINEERING V3 ---")
    
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
    
    logger.info("Beräknar Head-to-Head statistik...")
    df_with_h2h = _calculate_head_to_head(df_with_streaks)
    
    logger.info("Beräknar ligaposition...")
    df_with_position = _calculate_league_position(df_with_h2h)
    
    logger.info("Beräknar ELO-rating...")
    df_final = _calculate_elo(df_with_position)
    
    # Lägg till skade-features om tillgängligt
    if INJURY_FEATURES_AVAILABLE:
        logger.info("Lägger till skade-features...")
        df_final = _add_injury_features(df_final)

    # Säkerställ stabilt feature-kontrakt även om injury_scraper saknas
    for c, default in [
        ("InjuredPlayers_Home", 0), ("InjuredPlayers_Away", 0),
        ("KeyPlayersOut_Home", 0), ("KeyPlayersOut_Away", 0),
        ("InjurySeverity_Home", 0.0), ("InjurySeverity_Away", 0.0),
    ]:
        if c not in df_final.columns:
            df_final[c] = default
    
    added_cols = [c for c in df_final.columns if c not in df.columns]
    logger.info(f"Feature engineering slutförd. Nya kolumner ({len(added_cols)}): {added_cols}")
    
    return df_final


def _add_injury_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lägger till skade-features från injury_scraper
    
    Dessa features är statiska (inte historiska) och representerar
    aktuellt läge vid tidpunkten för prediktion.
    """
    # Initiera kolumner med default-värden
    df['InjuredPlayers_Home'] = 0
    df['InjuredPlayers_Away'] = 0
    df['KeyPlayersOut_Home'] = 0
    df['KeyPlayersOut_Away'] = 0
    df['InjurySeverity_Home'] = 0.0
    df['InjurySeverity_Away'] = 0.0
    
    # För historisk data sätter vi alla till 0
    # (vi har inte historisk skadedata)
    # När vi gör prediktioner uppdateras dessa i app.py
    
    logger.info("Skade-features initierade (kommer uppdateras vid prediktion)")
    return df
