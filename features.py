# features.py
from __future__ import annotations

from typing import Optional, Tuple
import pandas as pd


def compute_h2h(
    df_history: pd.DataFrame,
    home_team: str,
    away_team: str,
    asof_date: Optional[pd.Timestamp] = None,
    window: int = 5,
) -> Tuple[int, int, int, float]:
    """
    Beräkna H2H-statistik för rätt lagpar (home_team vs away_team).
    Returnerar: (home_wins, draws, away_wins, avg_home_goal_diff)
    där avg_home_goal_diff är med home_team-perspektiv (positivt = bra för home_team).

    Tar endast med matcher före asof_date om angivet.
    """
    if df_history.empty:
        return 0, 0, 0, 0.0

    df = df_history.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if asof_date is not None and "Date" in df.columns:
        df = df[df["Date"] < pd.to_datetime(asof_date)]

    mask = (
        ((df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team))
        | ((df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team))
    )
    h2h = df.loc[mask].sort_values("Date" if "Date" in df.columns else df.index, ascending=True)
    if h2h.empty:
        return 0, 0, 0, 0.0

    h2h = h2h.tail(window)

    home_wins = draws = away_wins = 0
    gd_sum = 0.0
    gd_n = 0

    for _, r in h2h.iterrows():
        ftr = r.get("FTR")
        fthg = r.get("FTHG", 0)
        ftag = r.get("FTAG", 0)
        ht = r.get("HomeTeam")
        at = r.get("AwayTeam")

        if pd.isna(ftr):
            continue

        # goal diff ur home_team-perspektiv
        if ht == home_team:
            gd = float(fthg) - float(ftag)
            if ftr == "H":
                home_wins += 1
            elif ftr == "D":
                draws += 1
            else:
                away_wins += 1
        else:
            gd = float(ftag) - float(fthg)  # home_team var bortalag här
            if ftr == "H":
                away_wins += 1
            elif ftr == "D":
                draws += 1
            else:
                home_wins += 1

        gd_sum += gd
        gd_n += 1

    avg_gd = (gd_sum / gd_n) if gd_n else 0.0
    return int(home_wins), int(draws), int(away_wins), float(avg_gd)
