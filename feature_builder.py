from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from schema import encode_league
from utils import normalize_team_name

import logging

logger = logging.getLogger(__name__)


def _mean(dq: deque) -> float:
    return float(np.mean(list(dq))) if len(dq) else 0.0


class FeatureBuilder:

    def __init__(self, k_factor: int = 20):
        self.k_factor = k_factor
        self._reset_state()

    def _reset_state(self) -> None:
        self._points_all: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self._gd_all: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self._points_home: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self._points_away: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self._goals_for: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self._goals_against: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))

        self._elo: Dict[str, float] = defaultdict(lambda: 1500.0)
        self._streak: Dict[str, int] = defaultdict(int)
        self._matches_played: Dict[str, int] = defaultdict(int)

        self._table: Dict[Tuple[int, str], Dict[str, Dict]] = defaultdict(
            lambda: defaultdict(lambda: {"points": 0, "gd": 0, "games": 0})
        )
        self._latest_league: Dict[str, int] = defaultdict(lambda: -1)
        self._latest_season: Dict[str, str] = defaultdict(lambda: "UNK")

        self._h2h: Dict[tuple, deque] = defaultdict(lambda: deque(maxlen=5))

        self._shots_for_5: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self._shots_against_5: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self._shots_for_10: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self._shots_against_10: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self._sot_for_5: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self._sot_against_5: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self._sot_for_10: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self._sot_against_10: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self._corners_for: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self._corners_against: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self._cards_5: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self._goals_total_5: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self._sot_total_5: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self._has_stats: Dict[str, bool] = defaultdict(bool)

    def _get_table_key(self, league_code: int, season: str) -> Tuple[int, str]:
        return (league_code, season)

    def _compute_position(self, league_code: int, season: str, team: str) -> int:
        tkey = self._get_table_key(league_code, season)
        tbl = self._table[tkey]
        if team not in tbl or tbl[team]["games"] == 0:
            return 0
        sorted_teams = sorted(
            tbl.items(),
            key=lambda x: (x[1]["points"], x[1]["gd"]),
            reverse=True,
        )
        for i, (t, _) in enumerate(sorted_teams, start=1):
            if t == team:
                return i
        return 0

    def _compute_h2h(self, home_team: str, away_team: str) -> Tuple[int, int, int, float]:
        matchup = tuple(sorted([home_team, away_team]))
        history = self._h2h[matchup]
        if not history:
            return 0, 0, 0, 0.0

        home_wins = draws = away_wins = 0
        gd_sum = 0.0
        gd_n = 0

        for prev_home, prev_away, prev_ftr, prev_gd in history:
            if prev_home == home_team:
                if prev_ftr == "H":
                    home_wins += 1
                elif prev_ftr == "D":
                    draws += 1
                else:
                    away_wins += 1
                gd_sum += prev_gd
            else:
                if prev_ftr == "H":
                    away_wins += 1
                elif prev_ftr == "D":
                    draws += 1
                else:
                    home_wins += 1
                gd_sum -= prev_gd
            gd_n += 1

        avg_gd = (gd_sum / gd_n) if gd_n else 0.0
        return home_wins, draws, away_wins, avg_gd

    def _stats_features(self, team: str, prefix: str) -> Dict[str, float]:
        eps = 1e-9
        has = 1.0 if self._has_stats[team] else 0.0
        f: Dict[str, float] = {}
        f[prefix + "Shots5"] = _mean(self._shots_for_5[team])
        f[prefix + "ShotsAg5"] = _mean(self._shots_against_5[team])
        f[prefix + "Shots10"] = _mean(self._shots_for_10[team])
        f[prefix + "ShotsAg10"] = _mean(self._shots_against_10[team])
        f[prefix + "SOT5"] = _mean(self._sot_for_5[team])
        f[prefix + "SOTAg5"] = _mean(self._sot_against_5[team])
        f[prefix + "SOT10"] = _mean(self._sot_for_10[team])
        f[prefix + "SOTAg10"] = _mean(self._sot_against_10[team])

        goals_sum = sum(self._goals_total_5[team]) if self._goals_total_5[team] else 0.0
        sot_sum = sum(self._sot_total_5[team]) if self._sot_total_5[team] else 0.0
        f[prefix + "Conversion"] = (goals_sum + 1.0) / (sot_sum + 2.0)

        corners_f = sum(self._corners_for[team]) if self._corners_for[team] else 0.0
        corners_a = sum(self._corners_against[team]) if self._corners_against[team] else 0.0
        f[prefix + "CornerShare"] = corners_f / (corners_f + corners_a + eps)

        f[prefix + "CardsRate"] = _mean(self._cards_5[team])
        return f

    def _pre_match_features(
        self, home_team: str, away_team: str, league_code: int, season: str
    ) -> Dict[str, float]:
        home_pos = self._compute_position(league_code, season, home_team)
        away_pos = self._compute_position(league_code, season, away_team)
        h2h_hw, h2h_d, h2h_aw, h2h_gd = self._compute_h2h(home_team, away_team)

        home_has = self._has_stats[home_team]
        away_has = self._has_stats[away_team]
        has_matchstats = 1.0 if (home_has and away_has) else 0.0

        home_sot_f = sum(self._sot_for_5[home_team]) if self._sot_for_5[home_team] else 0.0
        away_sot_f = sum(self._sot_for_5[away_team]) if self._sot_for_5[away_team] else 0.0
        eps = 1e-9
        sot_share_home = home_sot_f / (home_sot_f + away_sot_f + eps)

        features = {
            "HomeFormPts": _mean(self._points_all[home_team]),
            "HomeFormGD": _mean(self._gd_all[home_team]),
            "AwayFormPts": _mean(self._points_all[away_team]),
            "AwayFormGD": _mean(self._gd_all[away_team]),
            "HomeFormHome": _mean(self._points_home[home_team]),
            "AwayFormAway": _mean(self._points_away[away_team]),
            "HomeGoalsFor": _mean(self._goals_for[home_team]),
            "HomeGoalsAgainst": _mean(self._goals_against[home_team]),
            "AwayGoalsFor": _mean(self._goals_for[away_team]),
            "AwayGoalsAgainst": _mean(self._goals_against[away_team]),
            "HomeStreak": float(self._streak[home_team]),
            "AwayStreak": float(self._streak[away_team]),
            "H2H_HomeWins": float(h2h_hw),
            "H2H_Draws": float(h2h_d),
            "H2H_AwayWins": float(h2h_aw),
            "H2H_HomeGoalDiff": h2h_gd,
            "HomePosition": float(home_pos),
            "AwayPosition": float(away_pos),
            "PositionDiff": float(away_pos - home_pos),
            "HomeElo": float(self._elo[home_team]),
            "AwayElo": float(self._elo[away_team]),
            "League": float(league_code),
            "has_matchstats": has_matchstats,
            "SOTShareHome": sot_share_home,
        }
        features.update(self._stats_features(home_team, "Home"))
        features.update(self._stats_features(away_team, "Away"))
        return features

    def _update_state(
        self,
        home_team: str,
        away_team: str,
        fthg: int,
        ftag: int,
        ftr: str,
        league_code: int,
        season: str,
        stats: Optional[Dict[str, float]] = None,
    ) -> None:
        gd = fthg - ftag
        if ftr == "H":
            hp, ap = 3, 0
        elif ftr == "D":
            hp, ap = 1, 1
        else:
            hp, ap = 0, 3

        self._points_all[home_team].append(hp)
        self._points_all[away_team].append(ap)
        self._gd_all[home_team].append(gd)
        self._gd_all[away_team].append(-gd)

        self._points_home[home_team].append(hp)
        self._points_away[away_team].append(ap)

        self._goals_for[home_team].append(fthg)
        self._goals_against[home_team].append(ftag)
        self._goals_for[away_team].append(ftag)
        self._goals_against[away_team].append(fthg)

        def _do_streak(team: str, pts: int) -> None:
            if pts == 3:
                self._streak[team] = self._streak[team] + 1 if self._streak[team] >= 0 else 1
            elif pts == 0:
                self._streak[team] = self._streak[team] - 1 if self._streak[team] <= 0 else -1
            else:
                self._streak[team] = 0

        _do_streak(home_team, hp)
        _do_streak(away_team, ap)

        home_elo = float(self._elo[home_team])
        away_elo = float(self._elo[away_team])
        expected_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400.0))
        if ftr == "H":
            actual_home = 1.0
        elif ftr == "D":
            actual_home = 0.5
        else:
            actual_home = 0.0
        self._elo[home_team] = home_elo + self.k_factor * (actual_home - expected_home)
        self._elo[away_team] = away_elo + self.k_factor * ((1 - actual_home) - (1 - expected_home))

        tkey = self._get_table_key(league_code, season)
        tbl = self._table[tkey]
        if ftr == "H":
            tbl[home_team]["points"] += 3
        elif ftr == "D":
            tbl[home_team]["points"] += 1
            tbl[away_team]["points"] += 1
        else:
            tbl[away_team]["points"] += 3
        tbl[home_team]["gd"] += gd
        tbl[away_team]["gd"] -= gd
        tbl[home_team]["games"] += 1
        tbl[away_team]["games"] += 1

        matchup = tuple(sorted([home_team, away_team]))
        self._h2h[matchup].append((home_team, away_team, ftr, gd))

        self._matches_played[home_team] += 1
        self._matches_played[away_team] += 1
        self._latest_league[home_team] = league_code
        self._latest_league[away_team] = league_code
        self._latest_season[home_team] = season
        self._latest_season[away_team] = season

        if stats:
            hs = stats.get("HS", 0.0)
            as_ = stats.get("AS", 0.0)
            hst = stats.get("HST", 0.0)
            ast = stats.get("AST", 0.0)
            hc = stats.get("HC", 0.0)
            ac = stats.get("AC", 0.0)
            hy = stats.get("HY", 0.0)
            ay = stats.get("AY", 0.0)

            self._shots_for_5[home_team].append(hs)
            self._shots_against_5[home_team].append(as_)
            self._shots_for_10[home_team].append(hs)
            self._shots_against_10[home_team].append(as_)
            self._shots_for_5[away_team].append(as_)
            self._shots_against_5[away_team].append(hs)
            self._shots_for_10[away_team].append(as_)
            self._shots_against_10[away_team].append(hs)

            self._sot_for_5[home_team].append(hst)
            self._sot_against_5[home_team].append(ast)
            self._sot_for_10[home_team].append(hst)
            self._sot_against_10[home_team].append(ast)
            self._sot_for_5[away_team].append(ast)
            self._sot_against_5[away_team].append(hst)
            self._sot_for_10[away_team].append(ast)
            self._sot_against_10[away_team].append(hst)

            self._corners_for[home_team].append(hc)
            self._corners_against[home_team].append(ac)
            self._corners_for[away_team].append(ac)
            self._corners_against[away_team].append(hc)

            self._cards_5[home_team].append(hy)
            self._cards_5[away_team].append(ay)

            self._goals_total_5[home_team].append(float(fthg))
            self._sot_total_5[home_team].append(hst)
            self._goals_total_5[away_team].append(float(ftag))
            self._sot_total_5[away_team].append(ast)

            self._has_stats[home_team] = True
            self._has_stats[away_team] = True

    def fit(self, history_df: pd.DataFrame) -> pd.DataFrame:
        if history_df.empty:
            return pd.DataFrame()

        df = history_df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date", ascending=True).reset_index(drop=True)

        df["HomeTeam"] = df["HomeTeam"].apply(normalize_team_name)
        df["AwayTeam"] = df["AwayTeam"].apply(normalize_team_name)

        self._reset_state()

        feature_rows: List[Dict[str, float]] = []
        for _, row in df.iterrows():
            ht = row["HomeTeam"]
            at = row["AwayTeam"]
            fthg = int(row["FTHG"])
            ftag = int(row["FTAG"])
            ftr = row["FTR"]
            league_code = encode_league(row.get("League")) if "League" in df.columns else -1
            season = str(row.get("Season", "UNK")) if "Season" in df.columns else "UNK"

            stats = None
            if "HS" in df.columns:
                stat_vals = {}
                for sc in ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]:
                    if sc in df.columns:
                        v = row.get(sc)
                        stat_vals[sc] = float(v) if pd.notna(v) else 0.0
                stats = stat_vals if stat_vals else None

            features = self._pre_match_features(ht, at, league_code, season)
            odds_feat = self._odds_features(row, df.columns)
            features.update(odds_feat)
            feature_rows.append(features)
            self._update_state(ht, at, fthg, ftag, ftr, league_code, season, stats=stats)

        features_df = pd.DataFrame(feature_rows, index=df.index)
        for col in features_df.columns:
            df[col] = features_df[col]

        for c, default in [
            ("InjuredPlayers_Home", 0),
            ("InjuredPlayers_Away", 0),
            ("KeyPlayersOut_Home", 0),
            ("KeyPlayersOut_Away", 0),
            ("InjurySeverity_Home", 0.0),
            ("InjurySeverity_Away", 0.0),
        ]:
            if c not in df.columns:
                df[c] = default

        return df

    @staticmethod
    def _odds_features(row: pd.Series, columns: pd.Index) -> Dict[str, float]:
        _ODDS_PAIRS = [("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA")]
        for h_col, d_col, a_col in _ODDS_PAIRS:
            if h_col in columns and d_col in columns and a_col in columns:
                oh = row.get(h_col)
                od = row.get(d_col)
                oa = row.get(a_col)
                if pd.notna(oh) and pd.notna(od) and pd.notna(oa):
                    oh, od, oa = float(oh), float(od), float(oa)
                    if oh > 0 and od > 0 and oa > 0:
                        raw_h = 1.0 / oh
                        raw_d = 1.0 / od
                        raw_a = 1.0 / oa
                        total = raw_h + raw_d + raw_a
                        return {
                            "has_odds": 1.0,
                            "ImpliedHome": raw_h / total,
                            "ImpliedDraw": raw_d / total,
                            "ImpliedAway": raw_a / total,
                        }
        return {"has_odds": 0.0, "ImpliedHome": 1 / 3, "ImpliedDraw": 1 / 3, "ImpliedAway": 1 / 3}

    def features_for_match(
        self,
        home_team: str,
        away_team: str,
        league: Optional[str] = None,
        season: Optional[str] = None,
        odds: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[str, float]]:
        ht = normalize_team_name(home_team)
        at = normalize_team_name(away_team)

        if self._matches_played[ht] == 0 or self._matches_played[at] == 0:
            return None

        league_code = encode_league(league) if league else int(self._latest_league[ht])
        season_val = season if season else str(self._latest_season[ht])

        features = self._pre_match_features(ht, at, league_code, season_val)

        features["InjuredPlayers_Home"] = 0
        features["InjuredPlayers_Away"] = 0
        features["KeyPlayersOut_Home"] = 0
        features["KeyPlayersOut_Away"] = 0
        features["InjurySeverity_Home"] = 0.0
        features["InjurySeverity_Away"] = 0.0

        if odds:
            row_series = pd.Series(odds)
            odds_feat = self._odds_features(row_series, pd.Index(odds.keys()))
        else:
            odds_feat = {"has_odds": 0.0, "ImpliedHome": 1 / 3, "ImpliedDraw": 1 / 3, "ImpliedAway": 1 / 3}
        features.update(odds_feat)

        return features

    def get_team_state(self, team_name: str) -> Optional[Dict]:
        team = normalize_team_name(team_name)
        if self._matches_played[team] == 0:
            return None
        league_code = int(self._latest_league[team])
        season_val = str(self._latest_season[team])
        return {
            "FormPts": _mean(self._points_all[team]),
            "FormGD": _mean(self._gd_all[team]),
            "FormHome": _mean(self._points_home[team]),
            "FormAway": _mean(self._points_away[team]),
            "GoalsFor": _mean(self._goals_for[team]),
            "GoalsAgainst": _mean(self._goals_against[team]),
            "Streak": int(self._streak[team]),
            "Elo": float(self._elo[team]),
            "Position": int(self._compute_position(league_code, season_val, team)),
            "League": league_code,
            "MatchesPlayed": int(self._matches_played[team]),
        }

    def get_all_team_states(self) -> Dict[str, Dict]:
        states = {}
        all_teams = set(self._matches_played.keys())
        for team in all_teams:
            state = self.get_team_state(team)
            if state is not None:
                states[team] = state
        return states
