# schema.py
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

# Klass-mappning: Home/Draw/Away -> 0/1/2
CLASS_MAP: Dict[str, int] = {"H": 0, "D": 1, "A": 2}
INV_CLASS_MAP: Dict[int, str] = {0: "1", 1: "X", 2: "2"}

# Liga-koder (football-data.co.uk) -> numerisk kod (för modell)
LEAGUE_MAP: Dict[str, int] = {"E0": 0, "E1": 1, "E2": 2, "E3": 3}


def encode_league(value) -> int:
    if value is None:
        return -1
    s = str(value).strip()
    return LEAGUE_MAP.get(s, -1)


BASE_FEATURE_COLUMNS = [
    "HomeFormPts", "HomeFormGD", "AwayFormPts", "AwayFormGD",
    "HomeFormHome", "AwayFormAway",
    "HomeGoalsFor", "HomeGoalsAgainst", "AwayGoalsFor", "AwayGoalsAgainst",
    "HomeStreak", "AwayStreak",
    "H2H_HomeWins", "H2H_Draws", "H2H_AwayWins", "H2H_HomeGoalDiff",
    "HomePosition", "AwayPosition", "PositionDiff",
    "HomeElo", "AwayElo",
    "League",
    "InjuredPlayers_Home", "InjuredPlayers_Away",
    "KeyPlayersOut_Home", "KeyPlayersOut_Away",
    "InjurySeverity_Home", "InjurySeverity_Away",
]

STATS_FEATURE_COLUMNS = [
    "has_matchstats",
    "SOTShareHome",
    "HomeShots5", "HomeShotsAg5", "HomeShots10", "HomeShotsAg10",
    "HomeSOT5", "HomeSOTAg5", "HomeSOT10", "HomeSOTAg10",
    "HomeConversion", "HomeCornerShare", "HomeCardsRate",
    "AwayShots5", "AwayShotsAg5", "AwayShots10", "AwayShotsAg10",
    "AwaySOT5", "AwaySOTAg5", "AwaySOT10", "AwaySOTAg10",
    "AwayConversion", "AwayCornerShare", "AwayCardsRate",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + STATS_FEATURE_COLUMNS

ABLATION_GROUPS = {
    "base": BASE_FEATURE_COLUMNS,
    "+stats": BASE_FEATURE_COLUMNS + STATS_FEATURE_COLUMNS,
}


def proba_to_1x2(proba: Sequence[float], classes: Optional[Sequence[int]] = None) -> Dict[str, float]:
    """
    Returnerar sannolikheter som {"1": p_home, "X": p_draw, "2": p_away}.
    Om classes anges (t.ex. model.classes_), används den för robust mapping.
    """
    if classes is None:
        # Antag standardordning [0,1,2] = [H,D,A]
        return {"1": float(proba[0]), "X": float(proba[1]), "2": float(proba[2])}

    out = {"1": 0.0, "X": 0.0, "2": 0.0}
    for p, cls in zip(proba, classes):
        sign = INV_CLASS_MAP.get(int(cls))
        if sign:
            out[sign] = float(p)
    return out


def get_expected_feature_columns(model) -> list[str]:
    """
    Om modellen är tränad med feature_names_in_ (sklearn), använd den.
    Annars fall tillbaka till FEATURE_COLUMNS.
    """
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        return list(cols)
    return list(FEATURE_COLUMNS)
