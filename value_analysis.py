"""
value_analysis.py -- Value-lager ovanpa oddsverktyget.

Beraknar edge och expected return genom att jamfora en
bookmakers fair-sannolikheter mot en jamforelsesannolikhet
(comparison probability).

I version 1 anvands marknadskonsensus mellan bookmakers som
jamforelsekalla. Det innebar att comparison probability ar
det viktade snittet av alla tillgangliga bookmakers implicita
sannolikheter, normaliserat till summa 1.

Viktigt: positiv edge betyder att jamforelsesannolikheten ar
hogre an marknadens fair probability. Det ar inte en garanti
for vinst — det ar ett beslutsstod.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from odds_tool import (
    MatchOddsReport,
    OddsEntry,
    compute_overround,
    find_best_odds,
    odds_to_implied_probs,
    remove_overround,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanter
# ---------------------------------------------------------------------------

# Edge smaller than this (absolute value) is classified as neutral
EDGE_NEUTRAL_THRESHOLD = 0.01  # 1 percentage point


# ---------------------------------------------------------------------------
# Dataklasser
# ---------------------------------------------------------------------------

@dataclass
class OutcomeValue:
    """Value-analys for ett enskilt utfall (1, X eller 2)."""
    outcome: str  # "1", "X", or "2"
    odds: float
    bookmaker: str
    market_fair_prob: float
    comparison_prob: float
    edge: float
    expected_return: float
    edge_label: str  # "positiv edge", "neutral", "negativ edge"


@dataclass
class MatchValueReport:
    """Komplett value-rapport for en match."""
    home_team: str
    away_team: str
    outcomes: List[OutcomeValue]
    comparison_source: str  # t.ex. "marknadskonsensus (3 bookmakers)"
    num_bookmakers: int
    overround: float


# ---------------------------------------------------------------------------
# Karnfunktioner
# ---------------------------------------------------------------------------

def compute_edge(comparison_prob: float, market_fair_prob: float) -> float:
    """
    Beraknar edge (skillnad) mellan jamforelsesannolikhet och
    marknadens fair probability.

    edge = comparison_prob - market_fair_prob

    Positivt varde: jamforelsesannolikheten ar hogre an marknaden
    — potentiellt intressant.

    Negativt varde: marknaden ger hogre sannolikhet an jamforelsen.

    Returnerar edge som decimaltal (t.ex. 0.05 = 5 procentenheter).
    """
    return comparison_prob - market_fair_prob


def compute_expected_return(comparison_prob: float, odds: float) -> float:
    """
    Beraknar forvantad avkastning (expected return) for ett utfall.

    expected_return = comparison_prob * odds - 1

    Positivt varde innebar att jamforelsesannolikheten antyder att
    oddsen ar generosa. Negativt varde innebar det motsatta.

    Parametrar
    ----------
    comparison_prob : float
        Jamforelsesannolikhet (0-1).
    odds : float
        Decimalodds (> 1.0).

    Returnerar
    ----------
    float : forvantad avkastning (-1 till ...)
    """
    if odds <= 1.0:
        return -1.0
    return comparison_prob * odds - 1.0


def classify_edge(edge: float, threshold: float = EDGE_NEUTRAL_THRESHOLD) -> str:
    """
    Klassificerar edge som positiv, neutral eller negativ.

    Parametrar
    ----------
    edge : float
        Edge-varde (decimaltal).
    threshold : float
        Gransvarde for att klassificera som neutral.
        Standard: 0.01 (1 procentenhet).

    Returnerar
    ----------
    str : "positiv edge", "neutral", eller "negativ edge"
    """
    if edge > threshold:
        return "positiv edge"
    elif edge < -threshold:
        return "negativ edge"
    return "neutral"


# ---------------------------------------------------------------------------
# Marknadskonsensus
# ---------------------------------------------------------------------------

def compute_market_consensus(
    odds_entries: List[OddsEntry],
) -> Optional[Dict[str, float]]:
    """
    Beraknar marknadskonsensus-sannolikheter fran flera bookmakers.

    Metod: beraknar implicita sannolikheter for varje bookmaker,
    tar medelvarde per utfall, och normaliserar sa att summan = 1.

    Returnerar None om inga giltiga odds finns.

    Parametrar
    ----------
    odds_entries : List[OddsEntry]
        Lista av bookmaker-odds.

    Returnerar
    ----------
    Dict[str, float] eller None
        {"1": prob, "X": prob, "2": prob} normaliserade till 1.0.
    """
    if not odds_entries:
        return None

    all_implied: Dict[str, List[float]] = {"1": [], "X": [], "2": []}

    for entry in odds_entries:
        try:
            if entry.home > 1.0 and entry.draw > 1.0 and entry.away > 1.0:
                implied = odds_to_implied_probs(entry.home, entry.draw, entry.away)
                for key in ("1", "X", "2"):
                    all_implied[key].append(implied[key])
        except (ValueError, TypeError):
            continue

    if not all_implied["1"]:
        return None

    # Medelvarde per utfall
    avg: Dict[str, float] = {}
    for key in ("1", "X", "2"):
        avg[key] = sum(all_implied[key]) / len(all_implied[key])

    # Normalisera till summa 1
    return remove_overround(avg)


# ---------------------------------------------------------------------------
# Value-rapport
# ---------------------------------------------------------------------------

def build_value_report(
    report: MatchOddsReport,
    comparison_probs: Optional[Dict[str, float]] = None,
) -> Optional[MatchValueReport]:
    """
    Bygger en value-rapport for en match baserat pa oddsrapporten.

    Om comparison_probs inte anges, beraknas marknadskonsensus
    fran alla bookmakers i rapporten.

    Anvander basta tillgangliga odds per utfall for value-berakningen.

    Parametrar
    ----------
    report : MatchOddsReport
        Befintlig oddsrapport fran odds_tool.
    comparison_probs : Dict[str, float], optional
        Egna jamforelsesannolikheter {"1": ..., "X": ..., "2": ...}.
        Om None anvands marknadskonsensus.

    Returnerar
    ----------
    MatchValueReport eller None om value inte kan beraknas.
    """
    if not report.bookmaker_odds:
        return None

    num_bm = len(report.bookmaker_odds)

    # Bestam comparison probability
    if comparison_probs is not None:
        comp = comparison_probs
        source = "manuell jamforelsesannolikhet"
    else:
        if num_bm < 2:
            # Med bara en bookmaker anvands fair probs som bade
            # market och comparison — edge blir 0.
            # Returnera rapport med 0 edge for transparency.
            comp = report.fair_probs
            source = f"enda bookmaker ({report.bookmaker_odds[0].bookmaker})"
        else:
            consensus = compute_market_consensus(report.bookmaker_odds)
            if consensus is None:
                return None
            comp = consensus
            source = f"marknadskonsensus ({num_bm} bookmakers)"

    # Validera comparison probs
    for key in ("1", "X", "2"):
        if key not in comp:
            return None

    # Anvand basta tillgangliga odds per utfall
    best = find_best_odds(report.bookmaker_odds)
    if not best:
        return None

    # Market fair probs fran basta odds
    best_implied = {}
    for key in ("1", "X", "2"):
        if key not in best:
            return None
        best_implied[key] = 1.0 / best[key][0]

    market_fair = remove_overround(best_implied)

    outcomes: List[OutcomeValue] = []
    for outcome_key in ("1", "X", "2"):
        odds_val, bm_name = best[outcome_key]
        mfp = market_fair[outcome_key]
        cp = comp[outcome_key]
        edge = compute_edge(cp, mfp)
        er = compute_expected_return(cp, odds_val)
        label = classify_edge(edge)

        outcomes.append(OutcomeValue(
            outcome=outcome_key,
            odds=odds_val,
            bookmaker=bm_name,
            market_fair_prob=mfp,
            comparison_prob=cp,
            edge=edge,
            expected_return=er,
            edge_label=label,
        ))

    return MatchValueReport(
        home_team=report.home_team,
        away_team=report.away_team,
        outcomes=outcomes,
        comparison_source=source,
        num_bookmakers=num_bm,
        overround=report.overround,
    )


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_outcomes_by_edge(
    value_reports: List[MatchValueReport],
) -> List[Tuple[str, str, OutcomeValue]]:
    """
    Rangordnar alla utfall fran alla matcher efter edge (hogst forst).

    Returnerar en lista av (home_team vs away_team, outcome_label, OutcomeValue)
    sorterad fran hogst positiv edge till lagst.

    Parametrar
    ----------
    value_reports : List[MatchValueReport]
        Lista av value-rapporter.

    Returnerar
    ----------
    List[Tuple[str, str, OutcomeValue]]
        (match_label, outcome_label, OutcomeValue) sorterad efter edge desc.
    """
    outcome_labels = {"1": "Hemma", "X": "Oavgjort", "2": "Borta"}
    items: List[Tuple[str, str, OutcomeValue]] = []

    for vr in value_reports:
        match_label = f"{vr.home_team} vs {vr.away_team}"
        for ov in vr.outcomes:
            label = outcome_labels.get(ov.outcome, ov.outcome)
            items.append((match_label, label, ov))

    items.sort(key=lambda x: x[2].edge, reverse=True)
    return items


def rank_matches_by_interest(
    value_reports: List[MatchValueReport],
) -> List[MatchValueReport]:
    """
    Rangordnar matcher efter hur intressanta de ar (hogst absolut edge forst).

    En match ar mer intressant om den har ett utfall med stor positiv edge.
    Sorterar pa det hogsta positiva edge-vardet i matchen.

    Parametrar
    ----------
    value_reports : List[MatchValueReport]
        Lista av value-rapporter.

    Returnerar
    ----------
    List[MatchValueReport] sorterad fran mest intressant till minst.
    """
    def max_positive_edge(vr: MatchValueReport) -> float:
        if not vr.outcomes:
            return -999.0
        return max(ov.edge for ov in vr.outcomes)

    return sorted(value_reports, key=max_positive_edge, reverse=True)


# ---------------------------------------------------------------------------
# Textformatering
# ---------------------------------------------------------------------------

def format_value_report(vr: MatchValueReport) -> str:
    """Formaterar en MatchValueReport till lasbar text."""
    lines: List[str] = []
    lines.append(f"Match: {vr.home_team} vs {vr.away_team}")
    lines.append(f"Overround: {vr.overround:.1f}%")
    lines.append(f"Jamforelsekalla: {vr.comparison_source}")
    lines.append(f"Antal bookmakers: {vr.num_bookmakers}")
    lines.append("")

    outcome_labels = {"1": "Hemma", "X": "Oavgjort", "2": "Borta"}

    for ov in vr.outcomes:
        label = outcome_labels.get(ov.outcome, ov.outcome)
        lines.append(f"  {label} ({ov.outcome}):")
        lines.append(f"    Odds:              {ov.odds:.2f} ({ov.bookmaker})")
        lines.append(f"    Market fair prob:   {ov.market_fair_prob:.1%}")
        lines.append(f"    Comparison prob:    {ov.comparison_prob:.1%}")
        lines.append(f"    Edge:              {ov.edge:+.1%}")
        lines.append(f"    Expected return:   {ov.expected_return:+.1%}")
        lines.append(f"    Bedomning:         {ov.edge_label}")
        lines.append("")

    lines.append(
        "Obs: positiv edge innebar att jamforelsesannolikheten ar hogre "
        "an marknadens fair probability. Det ar ett beslutsstod, inte en garanti."
    )

    return "\n".join(lines)
