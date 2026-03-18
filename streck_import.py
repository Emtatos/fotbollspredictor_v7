"""
streck_import.py -- Automatisk inlasning av streckdata.

Lader in streckprocent (1/X/2) fran en CSV-fil, validerar,
normaliserar och matchar mot matcher i oddsverktyget.

Standardkalla: data/streck_data.csv

CSV-format (minst dessa kolumner):
    HomeTeam, AwayTeam, Streck1, StreckX, Streck2

Valfria kolumner:
    Date        -- matchdatum (YYYY-MM-DD) for battre matchning
    Source      -- kallmarkering (t.ex. "Svenska Spel")

Anvandning:
    from streck_import import auto_load_streck

    result = auto_load_streck()
    if result is not None:
        streck_lookup, status = result
        # streck_lookup: Dict[str, Dict[str, float]]
        #   nyckel = "HomeTeam_AwayTeam", varde = {"1": ..., "X": ..., "2": ...}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanter
# ---------------------------------------------------------------------------

DEFAULT_STRECK_PATH = Path("data") / "streck_data.csv"

REQUIRED_COLUMNS = {"HomeTeam", "AwayTeam", "Streck1", "StreckX", "Streck2"}
OPTIONAL_COLUMNS = {"Date", "Source"}


# ---------------------------------------------------------------------------
# Dataklasser
# ---------------------------------------------------------------------------

@dataclass
class StreckRecord:
    """En enskild streckrad efter validering och normalisering."""
    home_team: str
    away_team: str
    streck_1: float
    streck_x: float
    streck_2: float
    date: Optional[str] = None
    source: Optional[str] = None


@dataclass
class StreckImportStatus:
    """Status for en streckinlasning."""
    loaded: bool
    source_path: str
    total_rows: int
    valid_rows: int
    skipped_rows: int
    matched_count: int
    unmatched_count: int
    source_label: str
    errors: List[str]


# ---------------------------------------------------------------------------
# Inlasning
# ---------------------------------------------------------------------------

def load_streck_data(
    path: Optional[Path] = None,
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Laser in streckdata fran en CSV-fil.

    Parametrar
    ----------
    path : Path, optional
        Sokvag till CSV-fil. Standard: data/streck_data.csv

    Returnerar
    ----------
    (DataFrame, errors) -- DataFrame med radata eller None,
                           plus en lista av felmeddelanden.
    """
    if path is None:
        path = DEFAULT_STRECK_PATH

    errors: List[str] = []

    if not path.exists():
        errors.append(f"Streckfil hittades inte: {path}")
        return None, errors

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        errors.append(f"Kunde inte lasa CSV: {exc}")
        return None, errors

    if df.empty:
        errors.append("CSV-filen ar tom.")
        return None, errors

    return df, errors


def validate_streck_data(
    df: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Validerar att nodvandiga kolumner finns och att varje rad
    har tolkbara 1/X/2-streckvarden.

    Returnerar
    ----------
    (validated_df, errors) -- DataFrame med giltiga rader, eller None.
    """
    errors: List[str] = []

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(
            f"Saknade kolumner: {', '.join(sorted(missing))}. "
            f"Hittade: {', '.join(df.columns.tolist())}"
        )
        return None, errors

    valid_mask = pd.Series(True, index=df.index)

    for col in ["Streck1", "StreckX", "Streck2"]:
        numeric = pd.to_numeric(df[col], errors="coerce")
        invalid_idx = numeric.isna()
        if invalid_idx.any():
            n_invalid = int(invalid_idx.sum())
            errors.append(
                f"{n_invalid} rad(er) har ogiltigt varde i kolumn {col}."
            )
        valid_mask &= ~invalid_idx
        # Kontrollera icke-negativa
        neg_mask = numeric < 0
        if neg_mask.any():
            n_neg = int(neg_mask.sum())
            errors.append(
                f"{n_neg} rad(er) har negativt varde i kolumn {col}."
            )
        valid_mask &= ~neg_mask

    # Kontrollera att HomeTeam och AwayTeam inte ar tomma
    for col in ["HomeTeam", "AwayTeam"]:
        empty_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if empty_mask.any():
            n_empty = int(empty_mask.sum())
            errors.append(f"{n_empty} rad(er) saknar {col}.")
        valid_mask &= ~empty_mask

    validated = df.loc[valid_mask].copy()

    if validated.empty:
        errors.append("Inga giltiga rader efter validering.")
        return None, errors

    # Konvertera streckkolumner till float
    for col in ["Streck1", "StreckX", "Streck2"]:
        validated[col] = pd.to_numeric(validated[col], errors="coerce")

    return validated, errors


def normalize_streck_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normaliserar streckdata sa att 1/X/2 summerar till 100%.

    Om vardena ar i procentform (summa nara 100) behalles de
    som procent. Om de ar i decimalform (summa nara 1)
    konverteras de till procentform.

    Returnerar DataFrame med normaliserade streckvarden (procentform).
    """
    result = df.copy()

    s1 = result["Streck1"].values
    sx = result["StreckX"].values
    s2 = result["Streck2"].values
    totals = s1 + sx + s2

    for i in range(len(result)):
        total = totals[i]
        if total <= 0:
            continue
        if total <= 1.5:
            # Decimalform -- konvertera till procent
            result.iloc[i, result.columns.get_loc("Streck1")] = s1[i] * 100.0
            result.iloc[i, result.columns.get_loc("StreckX")] = sx[i] * 100.0
            result.iloc[i, result.columns.get_loc("Streck2")] = s2[i] * 100.0

    return result


# ---------------------------------------------------------------------------
# Matchning mot fixtures
# ---------------------------------------------------------------------------

def _normalize_team_for_matching(name: str) -> str:
    """
    Normaliserar ett lagnamn for matchning.

    Forsoker forst med repoets normalize_team_name om tillganglig,
    annars enkel lower/strip.
    """
    try:
        from utils import normalize_team_name
        return normalize_team_name(name)
    except ImportError:
        return name.strip()


def _make_match_key(home: str, away: str) -> str:
    """Skapar en nyckel for matchning: 'NormalizedHome_NormalizedAway'."""
    h = _normalize_team_for_matching(home)
    a = _normalize_team_for_matching(away)
    return f"{h}_{a}"


def match_streck_to_fixtures(
    streck_records: List[StreckRecord],
    fixture_keys: List[str],
) -> Tuple[Dict[str, Dict[str, float]], List[str], List[str]]:
    """
    Matchar streckdata mot en lista av fixture-nycklar.

    Parametrar
    ----------
    streck_records : List[StreckRecord]
        Validerade och normaliserade streckposter.
    fixture_keys : List[str]
        Lista av "HomeTeam_AwayTeam"-nycklar fran oddsverktygets matcher.

    Returnerar
    ----------
    (matched_lookup, matched_keys, unmatched_keys)
        matched_lookup: Dict[fixture_key] -> {"1": pct, "X": pct, "2": pct}
        matched_keys: fixture-nycklar som matchades
        unmatched_keys: fixture-nycklar som INTE matchades
    """
    # Bygg en lookup fran normaliserade strecknycklar till streckdata
    streck_by_normalized: Dict[str, StreckRecord] = {}
    for rec in streck_records:
        key = _make_match_key(rec.home_team, rec.away_team)
        streck_by_normalized[key] = rec

    # Bygg ocksa en case-insensitive fallback
    streck_lower: Dict[str, StreckRecord] = {}
    for rec in streck_records:
        key = f"{rec.home_team.strip().lower()}_{rec.away_team.strip().lower()}"
        streck_lower[key] = rec

    matched_lookup: Dict[str, Dict[str, float]] = {}
    matched_keys: List[str] = []
    unmatched_keys: List[str] = []

    for fkey in fixture_keys:
        # Forsta forsok: exakt normaliserad nyckel
        if fkey in streck_by_normalized:
            rec = streck_by_normalized[fkey]
            matched_lookup[fkey] = {
                "1": rec.streck_1,
                "X": rec.streck_x,
                "2": rec.streck_2,
            }
            matched_keys.append(fkey)
            continue

        # Andra forsok: normalisera fixture-nyckeln och sok
        parts = fkey.split("_", 1)
        if len(parts) == 2:
            norm_key = _make_match_key(parts[0], parts[1])
            if norm_key in streck_by_normalized:
                rec = streck_by_normalized[norm_key]
                matched_lookup[fkey] = {
                    "1": rec.streck_1,
                    "X": rec.streck_x,
                    "2": rec.streck_2,
                }
                matched_keys.append(fkey)
                continue

        # Tredje forsok: case-insensitive
        fkey_lower = fkey.strip().lower()
        if fkey_lower in streck_lower:
            rec = streck_lower[fkey_lower]
            matched_lookup[fkey] = {
                "1": rec.streck_1,
                "X": rec.streck_x,
                "2": rec.streck_2,
            }
            matched_keys.append(fkey)
            continue

        unmatched_keys.append(fkey)

    return matched_lookup, matched_keys, unmatched_keys


# ---------------------------------------------------------------------------
# Records fran DataFrame
# ---------------------------------------------------------------------------

def dataframe_to_records(df: pd.DataFrame) -> List[StreckRecord]:
    """
    Konverterar en validerad och normaliserad DataFrame till StreckRecord-lista.
    """
    records: List[StreckRecord] = []
    for _, row in df.iterrows():
        rec = StreckRecord(
            home_team=str(row["HomeTeam"]).strip(),
            away_team=str(row["AwayTeam"]).strip(),
            streck_1=float(row["Streck1"]),
            streck_x=float(row["StreckX"]),
            streck_2=float(row["Streck2"]),
            date=str(row["Date"]).strip() if "Date" in df.columns and pd.notna(row.get("Date")) else None,
            source=str(row["Source"]).strip() if "Source" in df.columns and pd.notna(row.get("Source")) else None,
        )
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Huvudfunktion: auto_load_streck
# ---------------------------------------------------------------------------

def auto_load_streck(
    path: Optional[Path] = None,
    fixture_keys: Optional[List[str]] = None,
) -> Tuple[Optional[Dict[str, Dict[str, float]]], StreckImportStatus]:
    """
    Automatisk inlasning av streckdata -- hela flodet i ett anrop.

    1. Laser in CSV fran path (standard: data/streck_data.csv)
    2. Validerar kolumner och varden
    3. Normaliserar format
    4. Matchar mot fixtures (om fixture_keys anges)
    5. Returnerar lookup + status

    Parametrar
    ----------
    path : Path, optional
        Sokvag till streck-CSV. Standard: data/streck_data.csv
    fixture_keys : List[str], optional
        Lista av "HomeTeam_AwayTeam"-nycklar att matcha mot.
        Om None gors ingen matchning (alla poster returneras).

    Returnerar
    ----------
    (streck_lookup_or_None, status) -- alltid en tuple.
    streck_lookup ar None om inlasning misslyckas.
    """
    if path is None:
        path = DEFAULT_STRECK_PATH

    all_errors: List[str] = []

    # 1. Lasa in
    df, load_errors = load_streck_data(path)
    all_errors.extend(load_errors)

    if df is None:
        status = StreckImportStatus(
            loaded=False,
            source_path=str(path),
            total_rows=0,
            valid_rows=0,
            skipped_rows=0,
            matched_count=0,
            unmatched_count=0,
            source_label="Ingen fil hittad",
            errors=all_errors,
        )
        return None, status

    total_rows = len(df)

    # 2. Validera
    validated, val_errors = validate_streck_data(df)
    all_errors.extend(val_errors)

    if validated is None:
        status = StreckImportStatus(
            loaded=False,
            source_path=str(path),
            total_rows=total_rows,
            valid_rows=0,
            skipped_rows=total_rows,
            matched_count=0,
            unmatched_count=0,
            source_label="Valideringsfel",
            errors=all_errors,
        )
        return None, status

    skipped = total_rows - len(validated)

    # 3. Normalisera
    normalized = normalize_streck_data(validated)

    # 4. Konvertera till records
    records = dataframe_to_records(normalized)

    # Bestam kalla
    sources = {r.source for r in records if r.source}
    if sources:
        source_label = ", ".join(sorted(sources))
    else:
        source_label = str(path.name)

    # 5. Matcha mot fixtures
    if fixture_keys is not None:
        matched_lookup, matched_keys, unmatched_keys = match_streck_to_fixtures(
            records, fixture_keys,
        )
        matched_count = len(matched_keys)
        unmatched_count = len(unmatched_keys)
    else:
        # Ingen matchning -- returnera alla som lookup
        matched_lookup = {}
        for rec in records:
            key = f"{rec.home_team}_{rec.away_team}"
            matched_lookup[key] = {
                "1": rec.streck_1,
                "X": rec.streck_x,
                "2": rec.streck_2,
            }
        matched_count = len(matched_lookup)
        unmatched_count = 0

    status = StreckImportStatus(
        loaded=True,
        source_path=str(path),
        total_rows=total_rows,
        valid_rows=len(validated),
        skipped_rows=skipped,
        matched_count=matched_count,
        unmatched_count=unmatched_count,
        source_label=source_label,
        errors=all_errors,
    )

    return matched_lookup, status
