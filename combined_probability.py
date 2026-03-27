# combined_probability.py
"""
Kombinerar odds, modell och streck till en viktad sannolikhet
för bättre halvgarderingsval.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from uncertainty import entropy_norm


# Vikter: odds väger mest (skarpast signal), modell komplement, streck minst
DEFAULT_WEIGHTS = {
    "odds": 0.50,
    "model": 0.35,
    "streck": 0.15,
}


@dataclass
class CombinedMatchProbability:
    """Kombinerad sannolikhet för en match."""
    home_team: str
    away_team: str
    prob_1: float
    prob_x: float
    prob_2: float
    entropy: float
    sources: Dict[str, bool]  # vilka signaler som fanns tillgängliga
    streck_delta_1: float = 0.0  # streck - fair_prob för hemma
    streck_delta_x: float = 0.0
    streck_delta_2: float = 0.0

    @property
    def probs(self) -> np.ndarray:
        return np.array([self.prob_1, self.prob_x, self.prob_2])


def odds_to_fair_probs(odds_1: float, odds_x: float, odds_2: float) -> np.ndarray:
    """
    Konverterar decimalodds till fair probabilities (overround borttagen).

    Parametrar
    ----------
    odds_1, odds_x, odds_2 : float
        Decimalodds (t.ex. 2.32, 3.35, 2.95).

    Returnerar
    ----------
    np.ndarray med [p_home, p_draw, p_away] som summerar till 1.0.
    """
    if odds_1 <= 1.0 or odds_x <= 1.0 or odds_2 <= 1.0:
        return np.array([1/3, 1/3, 1/3])
    raw = np.array([1.0 / odds_1, 1.0 / odds_x, 1.0 / odds_2])
    return raw / raw.sum()


def combine_probabilities(
    odds_probs: Optional[np.ndarray] = None,
    model_probs: Optional[np.ndarray] = None,
    streck_pcts: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Viktar ihop tillgängliga sannolikhetskällor.

    Om en källa saknas fördelas dess vikt proportionellt
    på de som finns. Om inga källor finns returneras uniform.

    Parametrar
    ----------
    odds_probs : ndarray, optional
        Fair probabilities från odds [p1, px, p2].
    model_probs : ndarray, optional
        Modellens sannolikheter [p1, px, p2].
    streck_pcts : ndarray, optional
        Streckfördelning [s1, sx, s2] i decimalform (0-1).
    weights : dict, optional
        Vikter per källa. Standard: DEFAULT_WEIGHTS.

    Returnerar
    ----------
    np.ndarray med [p1, px, p2] summerar till 1.0.
    """
    w = weights or DEFAULT_WEIGHTS.copy()

    sources = {
        "odds": odds_probs,
        "model": model_probs,
        "streck": streck_pcts,
    }

    # Filtrera bort saknade källor
    available = {k: v for k, v in sources.items() if v is not None}

    if not available:
        return np.array([1/3, 1/3, 1/3])

    # Fördela vikter proportionellt på tillgängliga källor
    total_weight = sum(w[k] for k in available)
    normalized_weights = {k: w[k] / total_weight for k in available}

    combined = np.zeros(3)
    for key, probs in available.items():
        combined += normalized_weights[key] * np.array(probs)

    # Säkerställ att det summerar till 1.0
    s = combined.sum()
    if s > 0:
        combined /= s

    return combined


def build_combined_match(
    home_team: str,
    away_team: str,
    odds_1: Optional[float] = None,
    odds_x: Optional[float] = None,
    odds_2: Optional[float] = None,
    model_probs: Optional[np.ndarray] = None,
    streck_1: Optional[float] = None,
    streck_x: Optional[float] = None,
    streck_2: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> CombinedMatchProbability:
    """
    Bygger kombinerad sannolikhet för en match givet alla tillgängliga data.

    Parametrar
    ----------
    home_team, away_team : str
    odds_1, odds_x, odds_2 : float, optional
        Decimalodds.
    model_probs : ndarray, optional
        Modellens [p1, px, p2].
    streck_1, streck_x, streck_2 : float, optional
        Streckprocent i heltal (t.ex. 63 för 63%).
    weights : dict, optional

    Returnerar
    ----------
    CombinedMatchProbability
    """
    # Odds → fair probs
    odds_probs = None
    if odds_1 and odds_x and odds_2:
        odds_probs = odds_to_fair_probs(odds_1, odds_x, odds_2)

    # Streck → decimalform
    streck_probs = None
    if streck_1 is not None and streck_x is not None and streck_2 is not None:
        streck_probs = np.array([streck_1 / 100.0, streck_x / 100.0, streck_2 / 100.0])
        s = streck_probs.sum()
        if s > 0:
            streck_probs /= s

    combined = combine_probabilities(
        odds_probs=odds_probs,
        model_probs=model_probs,
        streck_pcts=streck_probs,
        weights=weights,
    )

    # Beräkna streck-delta mot fair probs (för att identifiera överstreckat)
    fair = odds_probs if odds_probs is not None else combined
    sd1 = (streck_probs[0] - fair[0]) if streck_probs is not None else 0.0
    sdx = (streck_probs[1] - fair[1]) if streck_probs is not None else 0.0
    sd2 = (streck_probs[2] - fair[2]) if streck_probs is not None else 0.0

    sources = {
        "odds": odds_probs is not None,
        "model": model_probs is not None,
        "streck": streck_probs is not None,
    }

    return CombinedMatchProbability(
        home_team=home_team,
        away_team=away_team,
        prob_1=float(combined[0]),
        prob_x=float(combined[1]),
        prob_2=float(combined[2]),
        entropy=entropy_norm(combined[0], combined[1], combined[2]),
        sources=sources,
        streck_delta_1=float(sd1),
        streck_delta_x=float(sdx),
        streck_delta_2=float(sd2),
    )
