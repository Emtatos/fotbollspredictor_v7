# state.py
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Any

import numpy as np
import pandas as pd

from schema import encode_league


def _mean(dq: deque) -> float:
    return float(np.mean(list(dq))) if len(dq) else 0.0


def build_current_team_states(df_history: pd.DataFrame, k_factor: int = 20) -> Dict[str, Dict[str, Any]]:
    """
    Bygger "current state" per lag genom att replaya historiken i datumordning.
    Detta undviker stale pre-match features från senaste matchrad.

    Kräver kolumner: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR (League valfri).
    Return: team -> dict med:
      FormPts, FormGD, FormHome, FormAway, GoalsFor, GoalsAgainst, Streak, Elo, Position, League
    """
    if df_history is None or df_history.empty:
        return {}

    df = df_history.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date", ascending=True)
    else:
        df = df.reset_index(drop=True)

    # Rolling deques (senaste 5)
    points_all = defaultdict(lambda: deque(maxlen=5))
    gd_all = defaultdict(lambda: deque(maxlen=5))
    points_home = defaultdict(lambda: deque(maxlen=5))
    points_away = defaultdict(lambda: deque(maxlen=5))
    goals_for = defaultdict(lambda: deque(maxlen=5))
    goals_against = defaultdict(lambda: deque(maxlen=5))

    # Elo och streak
    elo = defaultdict(lambda: 1500.0)
    streak = defaultdict(int)

    # Matches played counter per team (for trust score)
    matches_played = defaultdict(int)

    # Tabellstatistik (poäng + gd)
    # per league_code (int) -> team -> stats
    table = defaultdict(lambda: defaultdict(lambda: {"points": 0, "gd": 0, "games": 0}))
    latest_league_for_team = defaultdict(lambda: -1)

    for _, r in df.iterrows():
        ht = r["HomeTeam"]
        at = r["AwayTeam"]
        fthg = int(r.get("FTHG", 0))
        ftag = int(r.get("FTAG", 0))
        ftr = r.get("FTR")
        league_code = encode_league(r.get("League")) if "League" in df.columns else -1

        latest_league_for_team[ht] = league_code
        latest_league_for_team[at] = league_code

        # Elo update
        home_elo_before = float(elo[ht])
        away_elo_before = float(elo[at])
        expected_home = 1.0 / (1.0 + 10 ** ((away_elo_before - home_elo_before) / 400.0))
        if ftr == "H":
            actual_home = 1.0
        elif ftr == "D":
            actual_home = 0.5
        else:
            actual_home = 0.0
        elo[ht] = home_elo_before + k_factor * (actual_home - expected_home)
        elo[at] = away_elo_before + k_factor * ((1 - actual_home) - (1 - expected_home))

        # points + gd (för form)
        gd = fthg - ftag
        if ftr == "H":
            hp, ap = 3, 0
        elif ftr == "D":
            hp, ap = 1, 1
        else:
            hp, ap = 0, 3

        points_all[ht].append(hp)
        points_all[at].append(ap)
        gd_all[ht].append(gd)
        gd_all[at].append(-gd)

        points_home[ht].append(hp)
        points_away[at].append(ap)

        goals_for[ht].append(fthg)
        goals_against[ht].append(ftag)
        goals_for[at].append(ftag)
        goals_against[at].append(fthg)

        # streak
        def _update_streak(team: str, pts: int):
            if pts == 3:
                if streak[team] >= 0:
                    streak[team] += 1
                else:
                    streak[team] = 1
            elif pts == 0:
                if streak[team] <= 0:
                    streak[team] -= 1
                else:
                    streak[team] = -1
            else:
                streak[team] = 0

        _update_streak(ht, hp)
        _update_streak(at, ap)

        # Update matches played counter
        matches_played[ht] += 1
        matches_played[at] += 1

        # table update (points+gd)
        t = table[league_code]
        if ftr == "H":
            t[ht]["points"] += 3
        elif ftr == "D":
            t[ht]["points"] += 1
            t[at]["points"] += 1
        else:
            t[at]["points"] += 3

        t[ht]["gd"] += gd
        t[at]["gd"] -= gd
        t[ht]["games"] += 1
        t[at]["games"] += 1

    # Beräkna positioner per liga (senaste snapshot)
    positions = {}
    for league_code, stats_by_team in table.items():
        sorted_teams = sorted(
            stats_by_team.items(),
            key=lambda x: (x[1]["points"], x[1]["gd"]),
            reverse=True,
        )
        for i, (team, _) in enumerate(sorted_teams, start=1):
            positions[(league_code, team)] = i

    # Bygg state dict
    states: Dict[str, Dict[str, Any]] = {}
    all_teams = set(list(points_all.keys()) + list(elo.keys()) + list(latest_league_for_team.keys()))
    for team in all_teams:
        league_code = latest_league_for_team[team]
        states[team] = {
            "FormPts": _mean(points_all[team]),
            "FormGD": _mean(gd_all[team]),
            "FormHome": _mean(points_home[team]),
            "FormAway": _mean(points_away[team]),
            "GoalsFor": _mean(goals_for[team]),
            "GoalsAgainst": _mean(goals_against[team]),
            "Streak": int(streak[team]),
            "Elo": float(elo[team]),
            "Position": int(positions.get((league_code, team), 0)),
            "League": int(league_code),
            "MatchesPlayed": int(matches_played[team]),
        }
    return states
