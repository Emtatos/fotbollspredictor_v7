"""
odds_tool.py -- Oddsverktyg: karna for 1X2-odds

Beraknar implicita sannolikheter fran bookmaker-odds,
overround, overround-justerade sannolikheter och
best-odds-jamforelse over flera bookmakers.

Designat som grund for framtida value- och tipset-logik.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataklasser
# ---------------------------------------------------------------------------

@dataclass
class OddsEntry:
    """1X2-odds fran en enskild bookmaker."""
    bookmaker: str
    home: float
    draw: float
    away: float


@dataclass
class MatchOddsReport:
    """Sammanstallning for en match: odds, sannolikheter, overround, best odds."""
    home_team: str
    away_team: str
    bookmaker_odds: List[OddsEntry]
    implied_probs: Dict[str, float] = field(default_factory=dict)
    overround: float = 0.0
    fair_probs: Dict[str, float] = field(default_factory=dict)
    best_odds: Dict[str, Tuple[float, str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Karnfunktioner
# ---------------------------------------------------------------------------

def odds_to_implied_probs(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
) -> Dict[str, float]:
    """
    Beraknar ra implicita sannolikheter fran 1X2 decimal-odds.

    Returnerar {"1": p_home, "X": p_draw, "2": p_away}.
    Summerar typiskt >1.0 (bookmaker-marginal).

    Kastar ValueError vid ogiltiga odds (<= 1.0).
    """
    for label, odds_val in [("home", home_odds), ("draw", draw_odds), ("away", away_odds)]:
        if not isinstance(odds_val, (int, float)):
            raise TypeError(f"Ogiltigt odds-varde for {label}: {odds_val!r}")
        if odds_val <= 1.0:
            raise ValueError(
                f"Odds for {label} maste vara > 1.0 (fick {odds_val})"
            )

    return {
        "1": 1.0 / home_odds,
        "X": 1.0 / draw_odds,
        "2": 1.0 / away_odds,
    }


def compute_overround(implied_probs: Dict[str, float]) -> float:
    """
    Beraknar total overround (bookmaker-marginal) i procent.

    overround = (sum(implied) - 1) * 100

    Returnerar t.ex. 5.3 for 5.3 % overround.
    """
    total = sum(implied_probs.values())
    return (total - 1.0) * 100.0


def remove_overround(
    implied_probs: Dict[str, float],
) -> Dict[str, float]:
    """
    Normaliserar implicita sannolikheter sa att de summerar till 1.0.

    Anvander enkel proportionell normalisering (multiplicative method).
    Returnerar {"1": ..., "X": ..., "2": ...} som summerar till 1.0.
    """
    total = sum(implied_probs.values())
    if total <= 0:
        raise ValueError("Summan av implicita sannolikheter maste vara > 0")
    return {k: v / total for k, v in implied_probs.items()}


# ---------------------------------------------------------------------------
# Best-odds logik
# ---------------------------------------------------------------------------

def find_best_odds(
    odds_entries: List[OddsEntry],
) -> Dict[str, Tuple[float, str]]:
    """
    Hittar basta tillgangliga odds per utfall (1, X, 2) over flera bookmakers.

    Returnerar {"1": (odds, bookmaker), "X": (odds, bookmaker), "2": (odds, bookmaker)}.

    Om listan ar tom returneras tomt dict.
    """
    if not odds_entries:
        return {}

    best: Dict[str, Tuple[float, str]] = {}

    for entry in odds_entries:
        for outcome, odds_val in [("1", entry.home), ("X", entry.draw), ("2", entry.away)]:
            if odds_val <= 1.0:
                continue
            if outcome not in best or odds_val > best[outcome][0]:
                best[outcome] = (odds_val, entry.bookmaker)

    return best


def combined_best_overround(best_odds: Dict[str, Tuple[float, str]]) -> float:
    """
    Beraknar overround for de basta oddsen sammanslagna.

    Om best-odds ger overround < 0 kan arbitrage finnas (ej fokus har).
    Returnerar overround i procent.
    """
    if len(best_odds) < 3:
        return 0.0
    implied = {k: 1.0 / v[0] for k, v in best_odds.items()}
    return compute_overround(implied)


# ---------------------------------------------------------------------------
# Rapport-generering
# ---------------------------------------------------------------------------

def build_match_report(
    home_team: str,
    away_team: str,
    odds_entries: List[OddsEntry],
) -> Optional[MatchOddsReport]:
    """
    Bygger en komplett oddsrapport for en match.

    Returnerar None om inga giltiga odds finns.
    """
    valid_entries = []
    for entry in odds_entries:
        try:
            if entry.home > 1.0 and entry.draw > 1.0 and entry.away > 1.0:
                valid_entries.append(entry)
        except (TypeError, AttributeError):
            logger.warning("Ogiltigt OddsEntry-objekt: %s", entry)

    if not valid_entries:
        return None

    # Anvand forsta bookmaker for primara sannolikheter
    primary = valid_entries[0]
    implied = odds_to_implied_probs(primary.home, primary.draw, primary.away)
    overround = compute_overround(implied)
    fair = remove_overround(implied)
    best = find_best_odds(valid_entries)

    return MatchOddsReport(
        home_team=home_team,
        away_team=away_team,
        bookmaker_odds=valid_entries,
        implied_probs=implied,
        overround=overround,
        fair_probs=fair,
        best_odds=best,
    )


# ---------------------------------------------------------------------------
# CSV-inlasning fran football-data.co.uk format
# ---------------------------------------------------------------------------

# Kanda bookmaker-kolumn-prefix i football-data.co.uk CSV:er
_BOOKMAKER_COLUMNS = {
    "B365": ("B365H", "B365D", "B365A", "Bet365"),
    "PS": ("PSH", "PSD", "PSA", "Pinnacle"),
    "BW": ("BWH", "BWD", "BWA", "bwin"),
    "IW": ("IWH", "IWD", "IWA", "Interwetten"),
    "WH": ("WHH", "WHD", "WHA", "William Hill"),
    "VC": ("VCH", "VCD", "VCA", "VC Bet"),
}


def extract_odds_from_row(
    row,
    columns,
) -> List[OddsEntry]:
    """
    Extraherar alla tillgangliga bookmaker-odds fran en CSV-rad.

    Parametrar
    ----------
    row : pd.Series eller dict-liknande
        En rad fran football-data.co.uk CSV.
    columns : kolumnlista (pd.Index, list, etc.)
        Tillgangliga kolumner.

    Returnerar
    ----------
    Lista av OddsEntry med giltiga odds.
    """
    import pandas as pd

    entries: List[OddsEntry] = []

    for prefix, (h_col, d_col, a_col, bm_name) in _BOOKMAKER_COLUMNS.items():
        if h_col in columns and d_col in columns and a_col in columns:
            try:
                h_val = row[h_col] if hasattr(row, '__getitem__') else getattr(row, h_col, None)
                d_val = row[d_col] if hasattr(row, '__getitem__') else getattr(row, d_col, None)
                a_val = row[a_col] if hasattr(row, '__getitem__') else getattr(row, a_col, None)

                if pd.notna(h_val) and pd.notna(d_val) and pd.notna(a_val):
                    h_f, d_f, a_f = float(h_val), float(d_val), float(a_val)
                    if h_f > 1.0 and d_f > 1.0 and a_f > 1.0:
                        entries.append(OddsEntry(
                            bookmaker=bm_name,
                            home=h_f,
                            draw=d_f,
                            away=a_f,
                        ))
            except (ValueError, TypeError, KeyError) as exc:
                logger.debug("Kunde inte lasa odds fran %s: %s", bm_name, exc)

    return entries


def build_reports_from_dataframe(df) -> List[MatchOddsReport]:
    """
    Bygger oddsrapporter for alla matcher i en DataFrame (football-data.co.uk-format).

    Kraver att DataFrame har kolumnerna HomeTeam, AwayTeam och
    minst en uppsattning bookmaker-odds (t.ex. B365H/B365D/B365A).

    Returnerar lista av MatchOddsReport (tomma matcher hoppas over).
    """
    reports: List[MatchOddsReport] = []

    for _, row in df.iterrows():
        home = str(row.get("HomeTeam", ""))
        away = str(row.get("AwayTeam", ""))
        if not home or not away:
            continue

        entries = extract_odds_from_row(row, df.columns)
        report = build_match_report(home, away, entries)
        if report is not None:
            reports.append(report)

    return reports


# ---------------------------------------------------------------------------
# Textformatering
# ---------------------------------------------------------------------------

def format_report(report: MatchOddsReport) -> str:
    """Formaterar en MatchOddsReport till lasbar text."""
    lines: List[str] = []
    lines.append(f"Match: {report.home_team} vs {report.away_team}")
    lines.append(f"Overround: {report.overround:.1f}%")
    lines.append("")

    # Bookmaker-odds
    lines.append("Bookmaker-odds:")
    for entry in report.bookmaker_odds:
        lines.append(
            f"  {entry.bookmaker:15s}  1: {entry.home:.2f}  X: {entry.draw:.2f}  2: {entry.away:.2f}"
        )
    lines.append("")

    # Implicita sannolikheter (ra)
    lines.append("Implicita sannolikheter (ra):")
    lines.append(
        f"  1: {report.implied_probs['1']:.1%}  "
        f"X: {report.implied_probs['X']:.1%}  "
        f"2: {report.implied_probs['2']:.1%}"
    )

    # Overround-justerade
    lines.append("Overround-justerade (fair):")
    lines.append(
        f"  1: {report.fair_probs['1']:.1%}  "
        f"X: {report.fair_probs['X']:.1%}  "
        f"2: {report.fair_probs['2']:.1%}"
    )
    lines.append("")

    # Best odds
    if report.best_odds:
        lines.append("Basta odds per utfall:")
        for outcome in ["1", "X", "2"]:
            if outcome in report.best_odds:
                odds_val, bm = report.best_odds[outcome]
                label = {"1": "Hemma", "X": "Oavgjort", "2": "Borta"}[outcome]
                lines.append(f"  {label} ({outcome}): {odds_val:.2f} ({bm})")

    return "\n".join(lines)
