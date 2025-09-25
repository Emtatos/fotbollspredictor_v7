# feature_engineering.py
import logging
from collections import defaultdict, deque

import numpy as np
import pandas as pd

# Importera vår hjälpfunktion
from utils import normalize_team_name


def _calculate_form(df: pd.DataFrame) -> pd.DataFrame:
    """Beräknar rullande 5-matchers form för poäng och målskillnad."""
    team_points = defaultdict(lambda: deque(maxlen=5))
    team_gd = defaultdict(lambda: deque(maxlen=5))

    df['HomeFormPts'] = np.nan
    df['HomeFormGD'] = np.nan
    df['AwayFormPts'] = np.nan
    df['AwayFormGD'] = np.nan

    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        # form före match
        if len(team_points[home_team]) > 0:
            df.loc[index, 'HomeFormPts'] = float(np.mean(list(team_points[home_team])))
            df.loc[index, 'HomeFormGD'] = float(np.mean(list(team_gd[home_team])))

        if len(team_points[away_team]) > 0:
            df.loc[index, 'AwayFormPts'] = float(np.mean(list(team_points[away_team])))
            df.loc[index, 'AwayFormGD'] = float(np.mean(list(team_gd[away_team])))

        # resultat & uppdatering
        home_gd = row['FTHG'] - row['FTAG']
        if row['FTR'] == 'H':
            home_pts, away_pts = 3, 0
        elif row['FTR'] == 'D':
            home_pts, away_pts = 1, 1
        else:  # 'A'
            home_pts, away_pts = 0, 3

        team_points[home_team].append(home_pts)
        team_points[away_team].append(away_pts)
        team_gd[home_team].append(home_gd)
        team_gd[away_team].append(-home_gd)

    df[['HomeFormPts', 'HomeFormGD', 'AwayFormPts', 'AwayFormGD']] = df[
        ['HomeFormPts', 'HomeFormGD', 'AwayFormPts', 'AwayFormGD']
    ].fillna(0)

    return df


def _calculate_elo(df: pd.DataFrame, k_factor: int = 20) -> pd.DataFrame:
    """Beräknar ELO-rating för varje lag baserat på matchhistorik."""
    elo_ratings = defaultdict(lambda: 1500.0)

    df['HomeElo'] = np.nan
    df['AwayElo'] = np.nan

    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        home_elo_before = float(elo_ratings[home_team])
        away_elo_before = float(elo_ratings[away_team])

        df.loc[index, 'HomeElo'] = home_elo_before
        df.loc[index, 'AwayElo'] = away_elo_before

        expected_home = 1.0 / (1.0 + 10 ** ((away_elo_before - home_elo_before) / 400.0))

        if row['FTR'] == 'H':
            actual_home = 1.0
        elif row['FTR'] == 'D':
            actual_home = 0.5
        else:  # 'A'
            actual_home = 0.0

        new_home_elo = home_elo_before + k_factor * (actual_home - expected_home)
        new_away_elo = away_elo_before + k_factor * ((1 - actual_home) - (1 - expected_home))

        elo_ratings[home_team] = new_home_elo
        elo_ratings[away_team] = new_away_elo

    df[['HomeElo', 'AwayElo']] = df[['HomeElo', 'AwayElo']].fillna(1500.0)
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orkestrerar skapandet av alla features för en given DataFrame med matchdata.

    Parametrar
    ----------
    df : pd.DataFrame
        En ren DataFrame (t.ex. från data_processing.normalize_csv_data).

    Returnerar
    -------
    pd.DataFrame
        DataFramen med tillagda kolumner för form och ELO.
    """
    if df.empty:
        logging.warning("create_features: Tom DataFrame angavs – returnerar tom.")
        return pd.DataFrame()

    df_copy = df.copy()
    df_copy.sort_values(by="Date", inplace=True, ascending=True)
    df_copy.reset_index(drop=True, inplace=True)

    # Normalisera lagnamn
    df_copy['HomeTeam'] = df_copy['HomeTeam'].apply(normalize_team_name)
    df_copy['AwayTeam'] = df_copy['AwayTeam'].apply(normalize_team_name)

    # Beräkna features
    df_with_form = _calculate_form(df_copy)
    df_with_elo = _calculate_elo(df_with_form)

    added_cols = [c for c in df_with_elo.columns if c not in df.columns]
    logging.info("Feature engineering slutförd. Nya kolumner: %s", added_cols)

    return df_with_elo
