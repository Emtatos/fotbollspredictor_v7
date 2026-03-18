"""
streck_analysis.py -- Streckjamforelse ovanpa oddsverktyget.

Jamfor marknadens fair probability (fran odds) mot folkets
streckprocent (1 / X / 2) for att identifiera over- och
understreckade utfall.

Grundmatt:
    delta = streckprocent - fair_market_probability
    positiv delta = overstreckad (folket satsar mer an marknaden antyder)
    negativ delta = understreckad (folket satsar mindre an marknaden antyder)

Understreckade utfall kan vara intressanta for tipset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from odds_tool import MatchOddsReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanter
# ---------------------------------------------------------------------------

# Delta smaller than this (absolute value) is classified as neutral
STRECK_NEUTRAL_THRESHOLD = 0.02  # 2 procentenheter


# ---------------------------------------------------------------------------
# Dataklasser
# ---------------------------------------------------------------------------

@dataclass
class OutcomeStreck:
    """Streckjamforelse for ett enskilt utfall (1, X eller 2)."""
    outcome: str               # "1", "X", or "2"
    fair_prob: float           # marknadens fair probability (0-1)
    streck_pct: float          # streckprocent (0-1)
    delta: float               # streck_pct - fair_prob
    label: str                 # "overstreckad", "neutral", "understreckad"


@dataclass
class MatchStreckReport:
    """Komplett streckjamforelse for en match."""
    home_team: str
    away_team: str
    outcomes: List[OutcomeStreck]
    max_abs_delta: float       # storsta absoluta delta i matchen


# ---------------------------------------------------------------------------
# Karnfunktioner
# ---------------------------------------------------------------------------

def compute_streck_delta(streck_pct: float, fair_prob: float) -> float:
    """
    Beraknar delta mellan streckprocent och fair market probability.

    delta = streck_pct - fair_prob

    Positivt delta: overstreckad (folket satsar mer an marknaden antyder).
    Negativt delta: understreckad (folket satsar mindre an marknaden antyder).

    Bada varden ska vara i decimalform (0-1), inte procentform.

    Parametrar
    ----------
    streck_pct : float
        Streckprocent i decimalform (t.ex. 0.45 for 45%).
    fair_prob : float
        Marknadens fair probability i decimalform (t.ex. 0.40 for 40%).

    Returnerar
    ----------
    float : delta (decimaltal, t.ex. 0.05 = 5 procentenheter).
    """
    return streck_pct - fair_prob


def classify_streck_value(
    delta: float,
    threshold: float = STRECK_NEUTRAL_THRESHOLD,
) -> str:
    """
    Klassificerar ett utfall baserat pa streckdelta.

    Parametrar
    ----------
    delta : float
        Delta-varde (decimaltal).
    threshold : float
        Gransvarde for neutralzon.
        Standard: 0.02 (2 procentenheter).

    Returnerar
    ----------
    str : "overstreckad", "neutral", eller "understreckad"
    """
    if delta > threshold:
        return "overstreckad"
    elif delta < -threshold:
        return "understreckad"
    return "neutral"


def _validate_streck_input(
    streck: Dict[str, float],
) -> Optional[Dict[str, float]]:
    """
    Validerar och normaliserar streckprocent-input.

    Accepterar varden i antingen procentform (0-100) eller
    decimalform (0-1). Om summan ar nara 100 tolkas det som
    procentform och konverteras till decimalform.

    Returnerar None om input ar ogiltig.
    """
    for key in ("1", "X", "2"):
        if key not in streck:
            logger.warning("Streckdata saknar utfall %s", key)
            return None

    values = [streck["1"], streck["X"], streck["2"]]

    # Kontrollera att alla varden ar numeriska och icke-negativa
    for i, v in enumerate(values):
        if not isinstance(v, (int, float)):
            logger.warning("Ogiltigt streckvarde: %r", v)
            return None
        if v < 0:
            logger.warning("Negativt streckvarde: %s", v)
            return None

    total = sum(values)

    if total <= 0:
        logger.warning("Streckprocentsumma ar 0 eller negativ")
        return None

    # Autodetektera format: om summa ar nara 100, tolka som procent
    if total > 1.5:
        # Procentform — konvertera till decimalform
        return {
            "1": streck["1"] / total,
            "X": streck["X"] / total,
            "2": streck["2"] / total,
        }

    # Redan decimalform — normalisera om det inte summerar exakt till 1
    if abs(total - 1.0) > 0.001:
        return {
            "1": streck["1"] / total,
            "X": streck["X"] / total,
            "2": streck["2"] / total,
        }

    return {"1": streck["1"], "X": streck["X"], "2": streck["2"]}


# ---------------------------------------------------------------------------
# Rapport-generering
# ---------------------------------------------------------------------------

def build_streck_report(
    home_team: str,
    away_team: str,
    fair_probs: Dict[str, float],
    streck_pcts: Dict[str, float],
    threshold: float = STRECK_NEUTRAL_THRESHOLD,
) -> Optional[MatchStreckReport]:
    """
    Bygger en komplett streckjamforelserapport for en match.

    Parametrar
    ----------
    home_team : str
        Hemmalag.
    away_team : str
        Bortalag.
    fair_probs : Dict[str, float]
        Fair market probabilities {"1": ..., "X": ..., "2": ...} (0-1).
    streck_pcts : Dict[str, float]
        Streckprocent {"1": ..., "X": ..., "2": ...}.
        Accepterar bade procentform (0-100) och decimalform (0-1).
    threshold : float
        Gransvarde for neutralzon.

    Returnerar
    ----------
    MatchStreckReport eller None om input ar ogiltig.
    """
    # Validera fair probs
    for key in ("1", "X", "2"):
        if key not in fair_probs:
            return None

    # Validera och normalisera streckdata
    normalized_streck = _validate_streck_input(streck_pcts)
    if normalized_streck is None:
        return None

    outcomes: List[OutcomeStreck] = []
    for outcome_key in ("1", "X", "2"):
        fp = fair_probs[outcome_key]
        sp = normalized_streck[outcome_key]
        delta = compute_streck_delta(sp, fp)
        label = classify_streck_value(delta, threshold)

        outcomes.append(OutcomeStreck(
            outcome=outcome_key,
            fair_prob=fp,
            streck_pct=sp,
            delta=delta,
            label=label,
        ))

    max_abs_delta = max(abs(o.delta) for o in outcomes) if outcomes else 0.0

    return MatchStreckReport(
        home_team=home_team,
        away_team=away_team,
        outcomes=outcomes,
        max_abs_delta=max_abs_delta,
    )


def build_streck_report_from_odds_report(
    odds_report: MatchOddsReport,
    streck_pcts: Dict[str, float],
    threshold: float = STRECK_NEUTRAL_THRESHOLD,
) -> Optional[MatchStreckReport]:
    """
    Bygger streckrapport direkt fran en MatchOddsReport.

    Anvander fair_probs fran oddsrapporten.
    """
    if not odds_report.fair_probs:
        return None

    return build_streck_report(
        home_team=odds_report.home_team,
        away_team=odds_report.away_team,
        fair_probs=odds_report.fair_probs,
        streck_pcts=streck_pcts,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_outcomes_by_streck_delta(
    streck_reports: List[MatchStreckReport],
) -> List[Tuple[str, str, OutcomeStreck]]:
    """
    Rangordnar alla utfall fran alla matcher efter streckdelta.

    Mest understreckade forst (mest negativ delta forst).

    Returnerar
    ----------
    List[Tuple[str, str, OutcomeStreck]]
        (match_label, outcome_label, OutcomeStreck) sorterad fran
        mest understreckad till mest overstreckad.
    """
    outcome_labels = {"1": "Hemma", "X": "Oavgjort", "2": "Borta"}
    items: List[Tuple[str, str, OutcomeStreck]] = []

    for sr in streck_reports:
        match_label = f"{sr.home_team} vs {sr.away_team}"
        for os_item in sr.outcomes:
            label = outcome_labels.get(os_item.outcome, os_item.outcome)
            items.append((match_label, label, os_item))

    # Sortera: mest negativ delta (mest understreckad) forst
    items.sort(key=lambda x: x[2].delta)
    return items


def rank_matches_by_streck_interest(
    streck_reports: List[MatchStreckReport],
) -> List[MatchStreckReport]:
    """
    Rangordnar matcher efter hur intressanta de ar for tipset.

    En match ar mer intressant ju storre den storsta absoluta
    streckavvikelsen ar. Sorteras fran mest intressant (storst
    avvikelse) till minst.

    Parametrar
    ----------
    streck_reports : List[MatchStreckReport]
        Lista av streckrapporter.

    Returnerar
    ----------
    List[MatchStreckReport] sorterad fran mest intressant till minst.
    """
    return sorted(
        streck_reports,
        key=lambda sr: sr.max_abs_delta,
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Textformatering
# ---------------------------------------------------------------------------

def format_streck_report(sr: MatchStreckReport) -> str:
    """Formaterar en MatchStreckReport till lasbar text."""
    lines: List[str] = []
    lines.append(f"Match: {sr.home_team} vs {sr.away_team}")
    lines.append("")

    outcome_labels = {"1": "Hemma", "X": "Oavgjort", "2": "Borta"}

    for os_item in sr.outcomes:
        label = outcome_labels.get(os_item.outcome, os_item.outcome)
        lines.append(f"  {label} ({os_item.outcome}):")
        lines.append(f"    Fair market prob:  {os_item.fair_prob:.1%}")
        lines.append(f"    Streckprocent:     {os_item.streck_pct:.1%}")
        lines.append(f"    Delta:             {os_item.delta:+.1%}")
        lines.append(f"    Bedomning:         {os_item.label}")
        lines.append("")

    lines.append(
        "Obs: understreckad innebar att streckandelen ar lagre an marknadens "
        "fair probability. Overstreckad innebar att streckandelen ar hogre. "
        "Detta ar beslutsstod, inte garanti."
    )

    return "\n".join(lines)
