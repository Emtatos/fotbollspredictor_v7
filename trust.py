# trust.py
"""
Trust score module for data coverage assessment.

Computes a trust score (0-100) based on data coverage:
- Form window (how many recent matches per team)
- History depth (total matches played per team)
- H2H sample size
- League presence
"""
from __future__ import annotations

from typing import Dict, Any, Tuple


def compute_trust_features(
    home_state: Dict[str, Any],
    away_state: Dict[str, Any],
    h2h_home_wins: int,
    h2h_draws: int,
    h2h_away_wins: int,
    league_code: int,
) -> Dict[str, Any]:
    """
    Compute trust-related features from available data.

    Args:
        home_state: Team state dict from build_current_team_states (must include MatchesPlayed)
        away_state: Team state dict from build_current_team_states (must include MatchesPlayed)
        h2h_home_wins: Number of H2H home wins from compute_h2h
        h2h_draws: Number of H2H draws from compute_h2h
        h2h_away_wins: Number of H2H away wins from compute_h2h
        league_code: Encoded league code (-1 if unknown)

    Returns:
        Dict with trust features:
            form_n_home: Form window size for home team (0-5)
            form_n_away: Form window size for away team (0-5)
            history_n_home: Total matches played by home team
            history_n_away: Total matches played by away team
            h2h_n: Total H2H matches
            league_ok: 1 if league is known, 0 otherwise
    """
    history_n_home = home_state.get("MatchesPlayed", 0) if home_state else 0
    history_n_away = away_state.get("MatchesPlayed", 0) if away_state else 0

    form_n_home = min(history_n_home, 5)
    form_n_away = min(history_n_away, 5)

    h2h_n = h2h_home_wins + h2h_draws + h2h_away_wins

    league_ok = 1 if league_code >= 0 else 0

    return {
        "form_n_home": form_n_home,
        "form_n_away": form_n_away,
        "history_n_home": history_n_home,
        "history_n_away": history_n_away,
        "h2h_n": h2h_n,
        "league_ok": league_ok,
    }


def trust_score(features: Dict[str, Any]) -> Tuple[int, str]:
    """
    Compute trust score and label from trust features.

    Scoring (max 100 points):
        - Form coverage: up to 30p
            min(form_n_home, 5) / 5 * 15 + min(form_n_away, 5) / 5 * 15
        - History depth: up to 40p
            min(history_n_home, 20) / 20 * 20 + min(history_n_away, 20) / 20 * 20
        - H2H sample: up to 20p
            min(h2h_n, 10) / 10 * 20
        - League present: 10p
            10 if league_ok else 0

    Labels:
        HIGH: score >= 70
        MED: 40 <= score < 70
        LOW: score < 40

    Args:
        features: Dict from compute_trust_features

    Returns:
        Tuple of (score: int, label: str)
    """
    form_n_home = features.get("form_n_home", 0)
    form_n_away = features.get("form_n_away", 0)
    history_n_home = features.get("history_n_home", 0)
    history_n_away = features.get("history_n_away", 0)
    h2h_n = features.get("h2h_n", 0)
    league_ok = features.get("league_ok", 0)

    form_score = (min(form_n_home, 5) / 5 * 15) + (min(form_n_away, 5) / 5 * 15)

    history_score = (min(history_n_home, 20) / 20 * 20) + (min(history_n_away, 20) / 20 * 20)

    h2h_score = min(h2h_n, 10) / 10 * 20

    league_score = 10 if league_ok else 0

    total = form_score + history_score + h2h_score + league_score
    score = int(round(total))

    if score >= 70:
        label = "HIGH"
    elif score >= 40:
        label = "MED"
    else:
        label = "LOW"

    return score, label
