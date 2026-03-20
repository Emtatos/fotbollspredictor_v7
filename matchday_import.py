"""
matchday_import.py -- Importflode for aktuell omgang.

Hanterar import av fixtures, odds och streckdata for en aktuell omgang.
Validerar, matchar och bygger importstatus for UI-visning.

Anvandning:
    from matchday_import import (
        parse_fixtures_csv,
        parse_odds_csv,
        parse_streck_csv,
        match_matchday_data,
        MatchdayImportStatus,
        generate_fixtures_template,
        generate_odds_template,
        generate_streck_template,
    )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from odds_tool import OddsEntry, build_match_report, MatchOddsReport
from value_analysis import build_value_report, MatchValueReport
from streck_analysis import (
    build_streck_report_from_odds_report,
    MatchStreckReport,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanter
# ---------------------------------------------------------------------------

FIXTURES_REQUIRED_COLUMNS = {"HomeTeam", "AwayTeam"}

# Recognized odds column sets (prefix -> (home_col, draw_col, away_col, label))
_ODDS_COLUMN_SETS = {
    "B365": ("B365H", "B365D", "B365A", "Bet365"),
    "PS": ("PSH", "PSD", "PSA", "Pinnacle"),
    "BW": ("BWH", "BWD", "BWA", "bwin"),
    "IW": ("IWH", "IWD", "IWA", "Interwetten"),
    "WH": ("WHH", "WHD", "WHA", "William Hill"),
    "VC": ("VCH", "VCD", "VCA", "VC Bet"),
}

# Simple 1/X/2 column names as fallback
_SIMPLE_ODDS_COLUMNS = ("Home", "Draw", "Away")

STRECK_REQUIRED_COLUMNS = {"HomeTeam", "AwayTeam", "Streck1", "StreckX", "Streck2"}


# ---------------------------------------------------------------------------
# Dataklasser
# ---------------------------------------------------------------------------

@dataclass
class MatchdayFixture:
    """En fixture (match) i aktuell omgang."""
    home_team: str
    away_team: str
    match_key: str  # "HomeTeam_AwayTeam" for matching


@dataclass
class MatchdayImportStatus:
    """Samlad importstatus for en omgang."""
    fixtures_count: int = 0
    odds_rows_loaded: int = 0
    streck_rows_loaded: int = 0
    odds_matched: int = 0
    streck_matched: int = 0
    fully_matched: int = 0
    unmatched_odds: List[str] = field(default_factory=list)
    unmatched_streck: List[str] = field(default_factory=list)
    fixtures_without_odds: List[str] = field(default_factory=list)
    fixtures_without_streck: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class MatchdayMatch:
    """En match med komplett eller partiell data."""
    home_team: str
    away_team: str
    match_key: str
    odds_entries: List[OddsEntry] = field(default_factory=list)
    streck: Optional[Dict[str, float]] = None
    odds_report: Optional[MatchOddsReport] = None
    value_report: Optional[MatchValueReport] = None
    streck_report: Optional[MatchStreckReport] = None
    has_odds: bool = False
    has_streck: bool = False


# ---------------------------------------------------------------------------
# Team name normalization for matching
# ---------------------------------------------------------------------------

def _normalize_for_key(name: str) -> str:
    """Normaliserar ett lagnamn for nyckelskapande."""
    try:
        from utils import normalize_team_name
        return normalize_team_name(name)
    except ImportError:
        return name.strip()


def _make_key(home: str, away: str) -> str:
    """Skapar matchningsnyckel fran hemma- och bortalag."""
    h = _normalize_for_key(home)
    a = _normalize_for_key(away)
    return f"{h}_{a}"


# ---------------------------------------------------------------------------
# Fixtures parsing
# ---------------------------------------------------------------------------

def parse_fixtures_csv(
    df: pd.DataFrame,
) -> Tuple[List[MatchdayFixture], List[str]]:
    """
    Parsar och validerar fixtures-data fran en DataFrame.

    Kraver kolumner: HomeTeam, AwayTeam.

    Returnerar (fixtures_list, errors).
    """
    errors: List[str] = []

    missing = FIXTURES_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(
            f"Saknade kolumner i fixtures: {', '.join(sorted(missing))}. "
            f"Hittade: {', '.join(df.columns.tolist())}"
        )
        return [], errors

    fixtures: List[MatchdayFixture] = []
    for idx, row in df.iterrows():
        raw_home = row.get("HomeTeam")
        raw_away = row.get("AwayTeam")
        home = str(raw_home).strip() if pd.notna(raw_home) else ""
        away = str(raw_away).strip() if pd.notna(raw_away) else ""

        if not home or not away or home == "nan" or away == "nan":
            errors.append(f"Rad {idx + 1}: tomt lagnamn (HomeTeam='{home}', AwayTeam='{away}').")
            continue

        key = _make_key(home, away)
        fixtures.append(MatchdayFixture(
            home_team=home,
            away_team=away,
            match_key=key,
        ))

    if not fixtures:
        errors.append("Inga giltiga fixtures hittades.")

    return fixtures, errors


# ---------------------------------------------------------------------------
# Text-based fixture parsing (paste match list)
# ---------------------------------------------------------------------------

# Regex: splits on " - ", " – ", " — ", "-", "–", "—", " vs ", " vs. "
_FIXTURE_LINE_RE = re.compile(
    r'\s*[-–—]\s*|\s+vs\.?\s+',
    re.IGNORECASE,
)


@dataclass
class ParseFixtureLinesResult:
    """Result of parsing pasted fixture text."""
    fixtures: List[MatchdayFixture]
    invalid_lines: List[str]
    total_lines: int
    blank_lines: int
    valid_count: int
    invalid_count: int


def parse_fixture_lines(text: str) -> ParseFixtureLinesResult:
    """
    Parsar inklistrad matchtext till fixtures.

    Stodjer rader som:
      - "Leeds United - Brentford"
      - "Leeds United – Brentford"
      - "Leeds United — Brentford"
      - "Leeds United vs Brentford"
      - "Leeds United vs. Brentford"

    Tomma rader ignoreras. Rader som inte kan tolkas rapporteras.

    Returnerar ParseFixtureLinesResult med fixtures, ogiltiga rader och statistik.
    """
    lines = text.split("\n")
    total_lines = len(lines)
    blank_lines = 0
    fixtures: List[MatchdayFixture] = []
    invalid_lines: List[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            blank_lines += 1
            continue

        parts = _FIXTURE_LINE_RE.split(line, maxsplit=1)

        if len(parts) != 2:
            invalid_lines.append(line)
            continue

        home_raw = parts[0].strip()
        away_raw = parts[1].strip()

        if not home_raw or not away_raw:
            invalid_lines.append(line)
            continue

        key = _make_key(home_raw, away_raw)
        fixtures.append(MatchdayFixture(
            home_team=home_raw,
            away_team=away_raw,
            match_key=key,
        ))

    return ParseFixtureLinesResult(
        fixtures=fixtures,
        invalid_lines=invalid_lines,
        total_lines=total_lines,
        blank_lines=blank_lines,
        valid_count=len(fixtures),
        invalid_count=len(invalid_lines),
    )


# ---------------------------------------------------------------------------
# Automatic odds fetching from football-data.co.uk CSV files
# ---------------------------------------------------------------------------

def _find_season_csvs(data_dir: Optional[Path] = None) -> List[Path]:
    """
    Hittar alla ligasasongens CSV-filer i data-mappen.

    Letar efter filer som matchar E0_2526.csv, E1_2425.csv etc.
    Returnerar lista sorterad med senaste sasong forst.
    """
    if data_dir is None:
        data_dir = Path("data")

    if not data_dir.exists():
        return []

    csv_pattern = re.compile(r'^(E\d)_(\d{4})\.csv$')
    found: List[Tuple[str, Path]] = []

    for p in data_dir.iterdir():
        if p.is_file():
            m = csv_pattern.match(p.name)
            if m:
                season_code = m.group(2)
                found.append((season_code, p))

    # Sort by season code descending (latest first)
    found.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in found]


def fetch_odds_for_fixtures(
    fixtures: List[MatchdayFixture],
    data_dir: Optional[Path] = None,
) -> Tuple[Dict[str, List[OddsEntry]], int, int, List[str]]:
    """
    Hamtar odds automatiskt fran football-data.co.uk CSV-filer i data-mappen.

    For varje fixture letas i de senaste CSV-filerna efter en match med
    samma hemma- och bortalag. Odds extraheras fran alla tillgangliga
    bookmaker-kolumner.

    Parametrar
    ----------
    fixtures : List[MatchdayFixture]
        Lista av fixtures att hamta odds for.
    data_dir : Path, optional
        Sokvag till data-mappen. Standard: data/

    Returnerar
    ----------
    (odds_by_key, matched_count, unmatched_count, unmatched_labels)
        odds_by_key: Dict[match_key] -> List[OddsEntry]
        matched_count: antal fixtures med hittade odds
        unmatched_count: antal fixtures utan odds
        unmatched_labels: lista av "Home vs Away" for omatchade fixtures
    """
    csv_files = _find_season_csvs(data_dir)

    if not csv_files:
        labels = [f"{f.home_team} vs {f.away_team}" for f in fixtures]
        return {}, 0, len(fixtures), labels

    # Load all CSVs, newest season first, and build a lookup of odds by
    # normalized (home, away) pair. Newest season data takes precedence.
    odds_rows: Dict[str, Tuple[pd.Series, pd.Index]] = {}

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.warning("Kunde inte lasa %s: %s", csv_path, exc)
            continue

        if "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
            continue

        for idx, row in df.iterrows():
            home_raw = str(row.get("HomeTeam", "")).strip()
            away_raw = str(row.get("AwayTeam", "")).strip()
            if not home_raw or not away_raw or home_raw == "nan" or away_raw == "nan":
                continue

            norm_key = _make_key(home_raw, away_raw)
            # Only keep the first (newest season) entry per match
            if norm_key not in odds_rows:
                odds_rows[norm_key] = (row, df.columns)

    # Now match fixtures against the odds lookup
    odds_by_key: Dict[str, List[OddsEntry]] = {}
    unmatched_labels: List[str] = []

    for fixture in fixtures:
        fkey = fixture.match_key

        # Try exact key, then lowercase fallback
        row_data = odds_rows.get(fkey)
        if row_data is None:
            row_data = odds_rows.get(fkey.lower()) if fkey.lower() != fkey else None
        if row_data is None:
            # Try re-normalizing fixture teams against the odds lookup
            norm_key = _make_key(fixture.home_team, fixture.away_team)
            row_data = odds_rows.get(norm_key)

        if row_data is not None:
            row, columns = row_data
            from odds_tool import extract_odds_from_row
            entries = extract_odds_from_row(row, columns)
            if entries:
                odds_by_key[fkey] = entries
                continue

        unmatched_labels.append(f"{fixture.home_team} vs {fixture.away_team}")

    matched_count = len(odds_by_key)
    unmatched_count = len(fixtures) - matched_count

    return odds_by_key, matched_count, unmatched_count, unmatched_labels


# ---------------------------------------------------------------------------
# Odds parsing
# ---------------------------------------------------------------------------

def _detect_odds_columns(df: pd.DataFrame) -> List[Tuple[str, str, str, str]]:
    """
    Detekterar vilka oddskolumner som finns i DataFramen.

    Returnerar lista av (home_col, draw_col, away_col, bookmaker_label).
    """
    found: List[Tuple[str, str, str, str]] = []

    # Check standard bookmaker column sets
    for prefix, (h_col, d_col, a_col, label) in _ODDS_COLUMN_SETS.items():
        if h_col in df.columns and d_col in df.columns and a_col in df.columns:
            found.append((h_col, d_col, a_col, label))

    # Check simple Home/Draw/Away columns as fallback
    if not found:
        h, d, a = _SIMPLE_ODDS_COLUMNS
        if h in df.columns and d in df.columns and a in df.columns:
            found.append((h, d, a, "Manuell"))

    return found


def parse_odds_csv(
    df: pd.DataFrame,
) -> Tuple[Dict[str, List[OddsEntry]], int, List[str]]:
    """
    Parsar odds-data fran en DataFrame.

    Kraver minst: HomeTeam, AwayTeam + minst en uppsattning oddskolumner.
    Stodjer bade football-data.co.uk-format och enkelt Home/Draw/Away-format.

    Returnerar (odds_by_key, total_valid_rows, errors).
    odds_by_key: Dict[match_key] -> List[OddsEntry]
    """
    errors: List[str] = []

    # Check for team columns
    if "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
        errors.append(
            f"Saknade kolumner: HomeTeam och/eller AwayTeam. "
            f"Hittade: {', '.join(df.columns.tolist())}"
        )
        return {}, 0, errors

    # Detect odds columns
    odds_cols = _detect_odds_columns(df)
    if not odds_cols:
        errors.append(
            "Inga kanda oddskolumner hittades. "
            "Forvantat format: B365H/B365D/B365A, PSH/PSD/PSA, "
            "eller Home/Draw/Away."
        )
        return {}, 0, errors

    odds_by_key: Dict[str, List[OddsEntry]] = {}
    valid_rows = 0

    for idx, row in df.iterrows():
        home = str(row.get("HomeTeam", "")).strip()
        away = str(row.get("AwayTeam", "")).strip()

        if not home or not away or home == "nan" or away == "nan":
            errors.append(f"Rad {idx + 1}: tomt lagnamn i odds-data.")
            continue

        key = _make_key(home, away)
        entries: List[OddsEntry] = []

        for h_col, d_col, a_col, bm_label in odds_cols:
            try:
                h_val = float(row[h_col])
                d_val = float(row[d_col])
                a_val = float(row[a_col])

                if pd.isna(h_val) or pd.isna(d_val) or pd.isna(a_val):
                    continue

                if h_val > 1.0 and d_val > 1.0 and a_val > 1.0:
                    entries.append(OddsEntry(
                        bookmaker=bm_label,
                        home=h_val,
                        draw=d_val,
                        away=a_val,
                    ))
            except (ValueError, TypeError, KeyError):
                continue

        if entries:
            odds_by_key[key] = entries
            valid_rows += 1
        else:
            errors.append(
                f"Rad {idx + 1}: inga giltiga odds for {home} vs {away}."
            )

    return odds_by_key, valid_rows, errors


# ---------------------------------------------------------------------------
# Streck parsing
# ---------------------------------------------------------------------------

def parse_streck_csv(
    df: pd.DataFrame,
) -> Tuple[Dict[str, Dict[str, float]], int, List[str]]:
    """
    Parsar streckdata fran en DataFrame.

    Kraver kolumner: HomeTeam, AwayTeam, Streck1, StreckX, Streck2.

    Returnerar (streck_by_key, total_valid_rows, errors).
    streck_by_key: Dict[match_key] -> {"1": pct, "X": pct, "2": pct}
    """
    errors: List[str] = []

    missing = STRECK_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(
            f"Saknade kolumner i streckdata: {', '.join(sorted(missing))}. "
            f"Hittade: {', '.join(df.columns.tolist())}"
        )
        return {}, 0, errors

    streck_by_key: Dict[str, Dict[str, float]] = {}
    valid_rows = 0

    for idx, row in df.iterrows():
        home = str(row.get("HomeTeam", "")).strip()
        away = str(row.get("AwayTeam", "")).strip()

        if not home or not away or home == "nan" or away == "nan":
            errors.append(f"Rad {idx + 1}: tomt lagnamn i streckdata.")
            continue

        try:
            s1 = float(row["Streck1"])
            sx = float(row["StreckX"])
            s2 = float(row["Streck2"])
        except (ValueError, TypeError):
            errors.append(
                f"Rad {idx + 1}: ogiltiga streckvarden for {home} vs {away}."
            )
            continue

        if s1 < 0 or sx < 0 or s2 < 0:
            errors.append(
                f"Rad {idx + 1}: negativa streckvarden for {home} vs {away}."
            )
            continue

        total = s1 + sx + s2
        if total <= 0:
            errors.append(
                f"Rad {idx + 1}: strecksumma ar 0 for {home} vs {away}."
            )
            continue

        key = _make_key(home, away)
        streck_by_key[key] = {"1": s1, "X": sx, "2": s2}
        valid_rows += 1

    return streck_by_key, valid_rows, errors


# ---------------------------------------------------------------------------
# Matchning och analys
# ---------------------------------------------------------------------------

def match_matchday_data(
    fixtures: List[MatchdayFixture],
    odds_by_key: Dict[str, List[OddsEntry]],
    streck_by_key: Dict[str, Dict[str, float]],
) -> Tuple[List[MatchdayMatch], MatchdayImportStatus]:
    """
    Matchar fixtures mot odds och streckdata.

    Bygger MatchdayMatch-objekt med tillganglig data och
    beraknar odds/value/streck-rapporter dar det ar mojligt.

    Returnerar (matches, status).
    """
    status = MatchdayImportStatus(
        fixtures_count=len(fixtures),
        odds_rows_loaded=len(odds_by_key),
        streck_rows_loaded=len(streck_by_key),
    )

    # Build normalized key lookup for odds and streck
    # Also build case-insensitive fallbacks
    odds_norm: Dict[str, List[OddsEntry]] = {}
    odds_lower: Dict[str, List[OddsEntry]] = {}
    for k, v in odds_by_key.items():
        odds_norm[k] = v
        odds_lower[k.lower()] = v

    streck_norm: Dict[str, Dict[str, float]] = {}
    streck_lower: Dict[str, Dict[str, float]] = {}
    for k, v in streck_by_key.items():
        streck_norm[k] = v
        streck_lower[k.lower()] = v

    # Track which odds/streck keys were matched
    matched_odds_keys: set = set()
    matched_streck_keys: set = set()

    matches: List[MatchdayMatch] = []

    for fixture in fixtures:
        fkey = fixture.match_key

        match = MatchdayMatch(
            home_team=fixture.home_team,
            away_team=fixture.away_team,
            match_key=fkey,
        )

        # Match odds
        if fkey in odds_norm:
            match.odds_entries = odds_norm[fkey]
            match.has_odds = True
            matched_odds_keys.add(fkey)
        elif fkey.lower() in odds_lower:
            match.odds_entries = odds_lower[fkey.lower()]
            match.has_odds = True
            matched_odds_keys.add(fkey)
        else:
            status.fixtures_without_odds.append(
                f"{fixture.home_team} vs {fixture.away_team}"
            )

        # Match streck
        if fkey in streck_norm:
            match.streck = streck_norm[fkey]
            match.has_streck = True
            matched_streck_keys.add(fkey)
        elif fkey.lower() in streck_lower:
            match.streck = streck_lower[fkey.lower()]
            match.has_streck = True
            matched_streck_keys.add(fkey)
        else:
            status.fixtures_without_streck.append(
                f"{fixture.home_team} vs {fixture.away_team}"
            )

        # Build reports if odds exist
        if match.has_odds and match.odds_entries:
            report = build_match_report(
                match.home_team, match.away_team, match.odds_entries,
            )
            if report is not None:
                match.odds_report = report
                match.value_report = build_value_report(report)

                if match.has_streck and match.streck:
                    match.streck_report = build_streck_report_from_odds_report(
                        report, match.streck,
                    )

        matches.append(match)

    # Count matches
    status.odds_matched = sum(1 for m in matches if m.has_odds)
    status.streck_matched = sum(1 for m in matches if m.has_streck)
    status.fully_matched = sum(
        1 for m in matches if m.has_odds and m.has_streck
    )

    # Detect unmatched odds rows (odds keys not matched to any fixture)
    all_fixture_keys = {f.match_key for f in fixtures}
    all_fixture_keys_lower = {f.match_key.lower() for f in fixtures}
    for ok in odds_by_key:
        if ok not in all_fixture_keys and ok.lower() not in all_fixture_keys_lower:
            status.unmatched_odds.append(ok)

    for sk in streck_by_key:
        if sk not in all_fixture_keys and sk.lower() not in all_fixture_keys_lower:
            status.unmatched_streck.append(sk)

    return matches, status


# ---------------------------------------------------------------------------
# CSV-mallar
# ---------------------------------------------------------------------------

FIXTURES_TEMPLATE_CSV = (
    "HomeTeam,AwayTeam\n"
    "Arsenal,Liverpool\n"
    "Man City,Chelsea\n"
    "Brighton,Newcastle\n"
    "Tottenham,Man United\n"
    "Fulham,Brentford\n"
    "Aston Villa,Everton\n"
    "Wolves,Crystal Palace\n"
    "Bournemouth,West Ham\n"
    "Nottingham Forest,Leicester\n"
    "Ipswich,Southampton\n"
)

ODDS_TEMPLATE_CSV = (
    "HomeTeam,AwayTeam,B365H,B365D,B365A\n"
    "Arsenal,Liverpool,2.10,3.40,3.60\n"
    "Man City,Chelsea,1.50,4.50,6.50\n"
    "Brighton,Newcastle,2.60,3.30,2.80\n"
    "Tottenham,Man United,2.30,3.50,3.10\n"
    "Fulham,Brentford,2.40,3.20,3.00\n"
    "Aston Villa,Everton,1.80,3.60,4.50\n"
    "Wolves,Crystal Palace,2.50,3.20,2.90\n"
    "Bournemouth,West Ham,2.20,3.30,3.40\n"
    "Nottingham Forest,Leicester,2.00,3.40,3.80\n"
    "Ipswich,Southampton,2.70,3.20,2.70\n"
)

ODDS_MULTI_BM_TEMPLATE_CSV = (
    "HomeTeam,AwayTeam,B365H,B365D,B365A,PSH,PSD,PSA\n"
    "Arsenal,Liverpool,2.10,3.40,3.60,2.15,3.35,3.65\n"
    "Man City,Chelsea,1.50,4.50,6.50,1.52,4.40,6.60\n"
)

ODDS_SIMPLE_TEMPLATE_CSV = (
    "HomeTeam,AwayTeam,Home,Draw,Away\n"
    "Arsenal,Liverpool,2.10,3.40,3.60\n"
    "Man City,Chelsea,1.50,4.50,6.50\n"
)

STRECK_TEMPLATE_CSV = (
    "HomeTeam,AwayTeam,Streck1,StreckX,Streck2\n"
    "Arsenal,Liverpool,45,28,27\n"
    "Man City,Chelsea,55,25,20\n"
    "Brighton,Newcastle,38,30,32\n"
    "Tottenham,Man United,42,28,30\n"
    "Fulham,Brentford,36,32,32\n"
    "Aston Villa,Everton,50,28,22\n"
    "Wolves,Crystal Palace,40,30,30\n"
    "Bournemouth,West Ham,35,30,35\n"
    "Nottingham Forest,Leicester,44,28,28\n"
    "Ipswich,Southampton,38,30,32\n"
)


def generate_fixtures_template() -> str:
    """Returnerar CSV-mallstrang for fixtures."""
    return FIXTURES_TEMPLATE_CSV


def generate_odds_template() -> str:
    """Returnerar CSV-mallstrang for odds."""
    return ODDS_TEMPLATE_CSV


def generate_streck_template() -> str:
    """Returnerar CSV-mallstrang for streck."""
    return STRECK_TEMPLATE_CSV
