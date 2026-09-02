# combined_probability.py
"""
Kombinerar odds, modell och streck till en viktad sannolikhet
för bättre halvgarderingsval.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


OddsTriple = Tuple[Optional[float], Optional[float], Optional[float]]
StreckTriple = Tuple[Optional[float], Optional[float], Optional[float]]


def build_combined_matches(
    matches: Sequence[Tuple[str, str]],
    odds: Sequence[Optional[OddsTriple]],
    streck: Sequence[Optional[StreckTriple]],
    model_probs: Sequence[Optional[np.ndarray]],
    weights: Optional[Dict[str, float]] = None,
) -> List[CombinedMatchProbability]:
    """
    Gemensam builder för kombinerade sannolikheter över en hel omgång.

    Alla vyer (Odds & Value, Flera Matcher) ska anropa denna funktion så att
    spikar, gain-ranking och halvgarderingstecken bygger på exakt samma
    objekt oavsett vilken sida som visar dem.

    Parametrar
    ----------
    matches : sekvens av (home_team, away_team)
    odds : sekvens av (odds_1, odds_x, odds_2) eller None per match
    streck : sekvens av (streck_1, streck_x, streck_2) i heltalsprocent
        eller None per match
    model_probs : sekvens av modellens [p1, px, p2] eller None per match
    weights : dict, optional

    Returnerar
    ----------
    Lista med CombinedMatchProbability i samma ordning som `matches`.
    """
    n = len(matches)
    if not (len(odds) == len(streck) == len(model_probs) == n):
        raise ValueError(
            "matches, odds, streck och model_probs måste ha samma längd "
            f"({n}, {len(odds)}, {len(streck)}, {len(model_probs)})"
        )

    combined: List[CombinedMatchProbability] = []
    for i, (home, away) in enumerate(matches):
        o = odds[i] or (None, None, None)
        s = streck[i] or (None, None, None)
        combined.append(build_combined_match(
            home_team=home,
            away_team=away,
            odds_1=o[0],
            odds_x=o[1],
            odds_2=o[2],
            model_probs=model_probs[i],
            streck_1=s[0],
            streck_x=s[1],
            streck_2=s[2],
            weights=weights,
        ))
    return combined


_SOURCE_LABELS = (("odds", "odds"), ("model", "modell"), ("streck", "streck"))


def describe_sources_used(
    combined: Sequence[CombinedMatchProbability],
    weights: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Beskriver vilka källor som faktiskt användes i en lista kombinerade matcher.

    Returnerar t.ex. ["odds (50%)", "modell (35%, 11/13 matcher)", "streck (15%)"].
    En källa som saknas för alla matcher utelämnas; en källa som saknas för
    några matcher får sin täckning angiven.
    """
    w = weights or DEFAULT_WEIGHTS
    n = len(combined)
    parts: List[str] = []
    for key, label in _SOURCE_LABELS:
        count = sum(1 for cm in combined if cm.sources.get(key))
        if count == 0:
            continue
        pct = f"{w[key]:.0%}"
        if count == n:
            parts.append(f"{label} ({pct})")
        else:
            parts.append(f"{label} ({pct}, {count}/{n} matcher)")
    return parts


def _odds_triple(entry) -> Optional[OddsTriple]:
    """(home, draw, away) från OddsEntry-objekt eller dict, None om ogiltigt."""
    if entry is None:
        return None
    try:
        if hasattr(entry, "home"):
            return float(entry.home), float(entry.draw), float(entry.away)
        if isinstance(entry, dict):
            return float(entry["home"]), float(entry["draw"]), float(entry["away"])
    except (KeyError, TypeError, ValueError):
        pass
    return None


def _streck_triple(streck: Optional[Dict[str, float]]) -> Optional[StreckTriple]:
    if not streck:
        return None
    return streck.get("1"), streck.get("X"), streck.get("2")


def combined_from_matchday_matches(
    matchday_matches: Sequence,
    model_probs: Sequence[Optional[np.ndarray]],
) -> List[CombinedMatchProbability]:
    """
    Odds & Value-vägen: bygger kombinerade sannolikheter från
    MatchdayMatch-objekt (kupongskanning/import) plus modellprediktioner.
    """
    matches = [(m.home_team, m.away_team) for m in matchday_matches]
    odds = [
        _odds_triple(m.odds_report.bookmaker_odds[0])
        if m.odds_report and m.odds_report.bookmaker_odds else None
        for m in matchday_matches
    ]
    streck = [
        _streck_triple(m.streck) if m.has_streck else None
        for m in matchday_matches
    ]
    return build_combined_matches(matches, odds, streck, model_probs)


def _lookup_case_insensitive(mapping: Dict[str, object], key: str):
    if key in mapping:
        return mapping[key]
    low = key.lower()
    for k, v in mapping.items():
        if k.lower() == low:
            return v
    return None


def combined_from_current_round(
    matches: Sequence[Tuple[str, str]],
    current_round: Optional[Dict],
    model_probs: Sequence[Optional[np.ndarray]],
    make_key,
) -> List[CombinedMatchProbability]:
    """
    Flera Matcher-vägen: bygger kombinerade sannolikheter från inklistrade
    matcher, odds/streck i `current_round` (session state) och
    modellprediktioner.

    `make_key(home, away)` ger nyckeln som odds/streck är lagrade under.
    """
    odds_by_key = (current_round or {}).get("odds") or {}
    streck_by_key = (current_round or {}).get("streck") or {}

    odds: List[Optional[OddsTriple]] = []
    streck: List[Optional[StreckTriple]] = []
    for home, away in matches:
        key = make_key(home, away)
        entries = _lookup_case_insensitive(odds_by_key, key)
        odds.append(_odds_triple(entries[0]) if entries else None)
        streck.append(_streck_triple(_lookup_case_insensitive(streck_by_key, key)))
    return build_combined_matches(matches, odds, streck, model_probs)
