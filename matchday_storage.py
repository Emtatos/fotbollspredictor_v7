"""
matchday_storage.py -- Spara och aterlas importerad omgangsdata.

Hanterar lokal persistens av matchday-data (fixtures, odds, streck)
sa att anvandaren slipper ladda upp filer pa nytt varje gang.

Lagringsformat: JSON-fil i data/saved_matchday.json.

Anvandning:
    from matchday_storage import (
        save_matchday_data,
        load_saved_matchday_data,
        clear_saved_matchday_data,
        get_saved_matchday_status,
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanter
# ---------------------------------------------------------------------------

DEFAULT_STORAGE_PATH = Path("data") / "saved_matchday.json"


# ---------------------------------------------------------------------------
# Dataklasser
# ---------------------------------------------------------------------------

@dataclass
class SavedMatchdayStatus:
    """Status for sparad omgangsdata."""
    exists: bool = False
    saved_at: Optional[str] = None
    match_count: int = 0
    has_odds: bool = False
    has_streck: bool = False
    odds_count: int = 0
    streck_count: int = 0
    source: str = ""  # "sparad" eller "nyimporterad"


# ---------------------------------------------------------------------------
# Intern serialisering
# ---------------------------------------------------------------------------

def _serialize_matchday_data(
    fixtures: List[Dict[str, str]],
    odds_by_key: Dict[str, List[Dict[str, Any]]],
    streck_by_key: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """Serialiserar omgangsdata till ett JSON-kompatibelt dict."""
    return {
        "version": 1,
        "saved_at": datetime.utcnow().isoformat(timespec="seconds"),
        "fixtures": fixtures,
        "odds_by_key": odds_by_key,
        "streck_by_key": streck_by_key,
        "meta": {
            "match_count": len(fixtures),
            "odds_count": len(odds_by_key),
            "streck_count": len(streck_by_key),
        },
    }


def _deserialize_matchday_data(
    raw: Dict[str, Any],
) -> Tuple[
    List[Dict[str, str]],
    Dict[str, List[Dict[str, Any]]],
    Dict[str, Dict[str, float]],
    Dict[str, Any],
]:
    """
    Deserialiserar sparad data.

    Returnerar (fixtures, odds_by_key, streck_by_key, meta).
    Kastar ValueError vid ogiltigt format.
    """
    version = raw.get("version")
    if version != 1:
        raise ValueError(f"Okand lagringsversion: {version}")

    fixtures = raw.get("fixtures")
    if not isinstance(fixtures, list):
        raise ValueError("Ogiltigt fixtures-format i sparad data.")

    odds_by_key = raw.get("odds_by_key")
    if not isinstance(odds_by_key, dict):
        raise ValueError("Ogiltigt odds-format i sparad data.")

    streck_by_key = raw.get("streck_by_key")
    if not isinstance(streck_by_key, dict):
        raise ValueError("Ogiltigt streck-format i sparad data.")

    meta = raw.get("meta", {})
    meta["saved_at"] = raw.get("saved_at", "")

    return fixtures, odds_by_key, streck_by_key, meta


# ---------------------------------------------------------------------------
# Konvertering: OddsEntry <-> dict
# ---------------------------------------------------------------------------

def _odds_entries_to_dicts(
    odds_by_key: Dict[str, list],
) -> Dict[str, List[Dict[str, Any]]]:
    """Konverterar OddsEntry-objekt till serialiserbara dicts."""
    result: Dict[str, List[Dict[str, Any]]] = {}
    for key, entries in odds_by_key.items():
        serialized: List[Dict[str, Any]] = []
        for entry in entries:
            if hasattr(entry, "bookmaker"):
                serialized.append({
                    "bookmaker": entry.bookmaker,
                    "home": float(entry.home),
                    "draw": float(entry.draw),
                    "away": float(entry.away),
                })
            elif isinstance(entry, dict):
                serialized.append(entry)
        result[key] = serialized
    return result


def _dicts_to_odds_entries(
    odds_by_key: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, list]:
    """Konverterar serialiserade dicts tillbaka till OddsEntry-objekt."""
    try:
        from odds_tool import OddsEntry
    except ImportError:
        return odds_by_key

    result: Dict[str, list] = {}
    for key, entries in odds_by_key.items():
        restored: list = []
        for d in entries:
            if isinstance(d, dict) and "bookmaker" in d:
                restored.append(OddsEntry(
                    bookmaker=d["bookmaker"],
                    home=float(d["home"]),
                    draw=float(d["draw"]),
                    away=float(d["away"]),
                ))
            else:
                restored.append(d)
        result[key] = restored
    return result


def _fixtures_from_objects(fixtures: list) -> List[Dict[str, str]]:
    """Konverterar MatchdayFixture-objekt till serialiserbara dicts."""
    result: List[Dict[str, str]] = []
    for f in fixtures:
        if hasattr(f, "home_team"):
            result.append({
                "home_team": f.home_team,
                "away_team": f.away_team,
                "match_key": f.match_key,
            })
        elif isinstance(f, dict):
            result.append(f)
    return result


def _dicts_to_fixtures(fixtures: List[Dict[str, str]]) -> list:
    """Konverterar serialiserade dicts tillbaka till MatchdayFixture-objekt."""
    try:
        from matchday_import import MatchdayFixture
    except ImportError:
        return fixtures

    result: list = []
    for d in fixtures:
        if isinstance(d, dict) and "home_team" in d:
            result.append(MatchdayFixture(
                home_team=d["home_team"],
                away_team=d["away_team"],
                match_key=d.get("match_key", ""),
            ))
        else:
            result.append(d)
    return result


# ---------------------------------------------------------------------------
# Publikt API
# ---------------------------------------------------------------------------

def save_matchday_data(
    fixtures: list,
    odds_by_key: Dict[str, list],
    streck_by_key: Dict[str, Dict[str, float]],
    path: Optional[Path] = None,
) -> bool:
    """
    Sparar importerad omgangsdata till lokal fil.

    Parametrar
    ----------
    fixtures : list
        Lista av MatchdayFixture-objekt eller dicts.
    odds_by_key : dict
        Dict[match_key] -> List[OddsEntry] eller List[dict].
    streck_by_key : dict
        Dict[match_key] -> {"1": float, "X": float, "2": float}.
    path : Path, optional
        Sokvag till lagringsplats. Standard: data/saved_matchday.json.

    Returnerar True om sparningen lyckades.
    """
    if path is None:
        path = DEFAULT_STORAGE_PATH

    try:
        fixtures_dicts = _fixtures_from_objects(fixtures)
        odds_dicts = _odds_entries_to_dicts(odds_by_key)

        payload = _serialize_matchday_data(
            fixtures_dicts, odds_dicts, streck_by_key,
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info("Omgangsdata sparad till %s (%d matcher).", path, len(fixtures_dicts))
        return True

    except Exception as exc:
        logger.error("Kunde inte spara omgangsdata: %s", exc)
        return False


def load_saved_matchday_data(
    path: Optional[Path] = None,
) -> Optional[Tuple[list, Dict[str, list], Dict[str, Dict[str, float]], Dict[str, Any]]]:
    """
    Laser in sparad omgangsdata fran lokal fil.

    Returnerar (fixtures, odds_by_key, streck_by_key, meta) eller None
    om ingen sparad data finns eller om den ar trasig.

    fixtures: List[MatchdayFixture]
    odds_by_key: Dict[match_key] -> List[OddsEntry]
    streck_by_key: Dict[match_key] -> {"1": float, "X": float, "2": float}
    meta: Dict med saved_at, match_count, odds_count, streck_count.
    """
    if path is None:
        path = DEFAULT_STORAGE_PATH

    if not path.exists():
        logger.debug("Ingen sparad omgangsdata: %s finns inte.", path)
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        fixtures_dicts, odds_dicts, streck_by_key, meta = _deserialize_matchday_data(raw)

        fixtures = _dicts_to_fixtures(fixtures_dicts)
        odds_by_key = _dicts_to_odds_entries(odds_dicts)

        return fixtures, odds_by_key, streck_by_key, meta

    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Sparad omgangsdata ar trasig (%s): %s", path, exc)
        return None
    except Exception as exc:
        logger.error("Ovantat fel vid inlasning av sparad data: %s", exc)
        return None


def clear_saved_matchday_data(
    path: Optional[Path] = None,
) -> bool:
    """
    Rensar sparad omgangsdata genom att ta bort filen.

    Returnerar True om filen raderades (eller inte fanns).
    """
    if path is None:
        path = DEFAULT_STORAGE_PATH

    try:
        if path.exists():
            path.unlink()
            logger.info("Sparad omgangsdata rensad: %s", path)
        return True
    except Exception as exc:
        logger.error("Kunde inte rensa sparad data: %s", exc)
        return False


def get_saved_matchday_status(
    path: Optional[Path] = None,
) -> SavedMatchdayStatus:
    """
    Returnerar status for sparad omgangsdata utan att ladda allt.

    Laser metadata fran filen utan att deserialisera hela datastrukturen.
    """
    if path is None:
        path = DEFAULT_STORAGE_PATH

    if not path.exists():
        return SavedMatchdayStatus(exists=False, source="ingen sparad data")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if raw.get("version") != 1:
            return SavedMatchdayStatus(exists=False, source="okand version")

        meta = raw.get("meta", {})
        match_count = meta.get("match_count", 0)
        odds_count = meta.get("odds_count", 0)
        streck_count = meta.get("streck_count", 0)
        saved_at = raw.get("saved_at", "")

        return SavedMatchdayStatus(
            exists=True,
            saved_at=saved_at,
            match_count=match_count,
            has_odds=odds_count > 0,
            has_streck=streck_count > 0,
            odds_count=odds_count,
            streck_count=streck_count,
            source="sparad",
        )

    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Kunde inte lasa status fran sparad data: %s", exc)
        return SavedMatchdayStatus(exists=False, source="trasig fil")
    except Exception as exc:
        logger.error("Ovantat fel vid statuslasning: %s", exc)
        return SavedMatchdayStatus(exists=False, source="fel")
