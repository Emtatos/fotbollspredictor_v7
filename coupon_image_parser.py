"""
coupon_image_parser.py -- Extraktion av matcher, streck och odds fran kupongbilder.

Anvander OpenAI Vision API (gpt-4o-mini) for att tolka en skarmbildskupong
och extrahera strukturerad data: lag, streckprocent och odds.

Anvandning:
    from coupon_image_parser import parse_coupon_image, CouponExtractionResult

    result = parse_coupon_image(image_bytes, "coupon.png")
    if result.rows:
        for row in result.rows:
            print(row.home_team, row.away_team, row.odds_1, row.streck_1)
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataklasser
# ---------------------------------------------------------------------------

@dataclass
class CouponRow:
    """En tolkad rad fran kupongbilden."""
    home_team: str
    away_team: str
    streck_1: Optional[float] = None
    streck_x: Optional[float] = None
    streck_2: Optional[float] = None
    odds_1: Optional[float] = None
    odds_x: Optional[float] = None
    odds_2: Optional[float] = None
    confidence: str = "ok"  # "ok", "uncertain", "incomplete"
    notes: str = ""


@dataclass
class CouponExtractionResult:
    """Resultat av kupongbildstolkning."""
    rows: List[CouponRow]
    total_rows: int
    complete_rows: int
    uncertain_rows: int
    incomplete_rows: int
    raw_response: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Intern: bildkodning
# ---------------------------------------------------------------------------

def _encode_image_to_base64(image_bytes: bytes) -> str:
    """Kodar bildbytes till base64-strang."""
    return base64.b64encode(image_bytes).decode("utf-8")


def _detect_mime_type(filename: str) -> str:
    """Detekterar MIME-typ fran filnamn."""
    ext = Path(filename).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    return mime_map.get(ext, "image/png")


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def is_supported_image(filename: str) -> bool:
    """Kontrollerar om filnamnet har ett stott bildformat."""
    ext = Path(filename).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Intern: prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """Du ar en OCR-expert som tolkar svenska kupongbilder (tipsbilder).

Bilden visar en kupong med fotbollsmatcher. Kupongen kan vara fran
Stryktipset (13 matcher), Europatipset (variabelt antal) eller liknande.

For varje match, extrahera:
- HomeTeam (hemmalag)
- AwayTeam (bortalag)
- Streck1 (streckprocent for 1/hemma, i procent t.ex. 45)
- StreckX (streckprocent for X/oavgjort)
- Streck2 (streckprocent for 2/borta)
- Odds1 (odds for 1/hemma, decimalodds t.ex. 2.10)
- OddsX (odds for X/oavgjort)
- Odds2 (odds for 2/borta)
- confidence: "ok" om du ar saker, "uncertain" om text ar svarlast eller pixlad, "incomplete" om data saknas
- notes: kort kommentar om nagot var svartolkat (annars tom strang)

VIKTIGT:
- Returnera ENBART en giltig JSON-array. Ingen markdown, inga code fences, ingen text fore eller efter.
- Streckvarden ska vara i procentform (t.ex. 45 for 45%). Summan av Streck1+StreckX+Streck2 ska vara nara 100.
- Odds ska vara decimalodds (t.ex. 2.10).
- Om du inte kan lasa ett varde, satt det till null.
- Gissa INTE aggressivt. Om text ar svarlast eller pixlad, satt confidence till "uncertain".
- Om en hel rad ar olasbar, inkludera den med confidence "incomplete".
- Lagnamn ska vara sa exakta som mojligt fran bilden.

Exempel pa forvantad output:
[
  {
    "HomeTeam": "Arsenal",
    "AwayTeam": "Liverpool",
    "Streck1": 45,
    "StreckX": 28,
    "Streck2": 27,
    "Odds1": 2.10,
    "OddsX": 3.40,
    "Odds2": 3.60,
    "confidence": "ok",
    "notes": ""
  }
]
"""


# ---------------------------------------------------------------------------
# Karnfunktion: tolka kupongbild
# ---------------------------------------------------------------------------

def parse_coupon_image(
    image_bytes: bytes,
    filename: str = "coupon.png",
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> CouponExtractionResult:
    """
    Tolkar en kupongbild och extraherar matcher, streck och odds.

    Anvander OpenAI Vision API for att lasa bilden och returnera
    strukturerad data.

    Parametrar
    ----------
    image_bytes : bytes
        Rabildbytes (PNG, JPG, WEBP).
    filename : str
        Filnamn for MIME-typ-detektion.
    api_key : str, optional
        OpenAI API-nyckel. Om None, forsoks fran miljovariabler.
    model : str
        OpenAI-modell att anvanda. Standard: gpt-4o-mini.

    Returnerar
    ----------
    CouponExtractionResult med tolkade rader och statistik.
    """
    import os

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass

    if not api_key:
        return CouponExtractionResult(
            rows=[],
            total_rows=0,
            complete_rows=0,
            uncertain_rows=0,
            incomplete_rows=0,
            error="Ingen OpenAI API-nyckel tillganglig. Satt OPENAI_API_KEY.",
        )

    if not is_supported_image(filename):
        return CouponExtractionResult(
            rows=[],
            total_rows=0,
            complete_rows=0,
            uncertain_rows=0,
            incomplete_rows=0,
            error=f"Bildformatet stods inte: {Path(filename).suffix}. "
                  f"Stodda format: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # Koda bilden
    b64_image = _encode_image_to_base64(image_bytes)
    mime_type = _detect_mime_type(filename)

    # Anropa OpenAI Vision API med retry-logik
    max_retries = 2
    last_error: Optional[Exception] = None

    try:
        from openai import OpenAI
    except ImportError:
        return CouponExtractionResult(
            rows=[],
            total_rows=0,
            complete_rows=0,
            uncertain_rows=0,
            incomplete_rows=0,
            error="OpenAI-paketet ar inte installerat. Kor: pip install openai",
        )

    client = OpenAI(api_key=api_key)
    raw_text = ""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _EXTRACTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{b64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=4000,
            )

            raw_text = response.choices[0].message.content.strip()
            last_error = None
            break  # Lyckat anrop
        except Exception as e:
            last_error = e
            logger.warning(
                "OpenAI Vision API-anrop misslyckades (forsok %d/%d): %s",
                attempt,
                max_retries,
                e,
            )

    if last_error is not None:
        logger.error("OpenAI Vision API-anrop misslyckades efter %d forsok: %s", max_retries, last_error)
        return CouponExtractionResult(
            rows=[],
            total_rows=0,
            complete_rows=0,
            uncertain_rows=0,
            incomplete_rows=0,
            raw_response="",
            error=f"OpenAI API-fel efter {max_retries} forsok: {last_error}",
        )

    # Parsa JSON fran svar
    return _parse_extraction_response(raw_text)


# ---------------------------------------------------------------------------
# Intern: parsa API-svar
# ---------------------------------------------------------------------------

def _parse_extraction_response(raw_text: str) -> CouponExtractionResult:
    """
    Parsar JSON-svaret fran OpenAI till CouponExtractionResult.

    Hanterar ogiltigt JSON defensivt.
    """
    # Forsok extrahera JSON fran svaret (kan vara inbaddat i markdown)
    json_text = raw_text
    # Prova ta bort eventuella markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw_text)
    if json_match:
        json_text = json_match.group(1).strip()

    # Prova hitta en JSON-array direkt
    if not json_text.startswith("["):
        bracket_start = json_text.find("[")
        if bracket_start >= 0:
            json_text = json_text[bracket_start:]

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.error("Kunde inte parsa JSON fran Vision API: %s", e)
        return CouponExtractionResult(
            rows=[],
            total_rows=0,
            complete_rows=0,
            uncertain_rows=0,
            incomplete_rows=0,
            raw_response=raw_text,
            error=f"Kunde inte tolka API-svaret som JSON: {e}",
        )

    if not isinstance(parsed, list):
        return CouponExtractionResult(
            rows=[],
            total_rows=0,
            complete_rows=0,
            uncertain_rows=0,
            incomplete_rows=0,
            raw_response=raw_text,
            error="API-svaret ar inte en lista.",
        )

    rows = _json_to_coupon_rows(parsed)
    return _build_result(rows, raw_text)


def _safe_float(value: Any) -> Optional[float]:
    """Konverterar till float om mojligt, annars None."""
    if value is None:
        return None
    try:
        f = float(value)
        if f < 0:
            return None
        return f
    except (ValueError, TypeError):
        return None


def _safe_odds(value: Any) -> Optional[float]:
    """Konverterar till odds (float > 1.0) om mojligt, annars None."""
    f = _safe_float(value)
    if f is not None and f > 1.0:
        return f
    return None


def _json_to_coupon_rows(data: List[Dict[str, Any]]) -> List[CouponRow]:
    """Konverterar en lista av JSON-dicts till CouponRow-objekt."""
    rows: List[CouponRow] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        home = str(item.get("HomeTeam", "")).strip()
        away = str(item.get("AwayTeam", "")).strip()

        if not home and not away:
            continue

        row = CouponRow(
            home_team=home,
            away_team=away,
            streck_1=_safe_float(item.get("Streck1")),
            streck_x=_safe_float(item.get("StreckX")),
            streck_2=_safe_float(item.get("Streck2")),
            odds_1=_safe_odds(item.get("Odds1")),
            odds_x=_safe_odds(item.get("OddsX")),
            odds_2=_safe_odds(item.get("Odds2")),
            confidence=str(item.get("confidence", "ok")).strip().lower(),
            notes=str(item.get("notes", "")).strip(),
        )

        # Revalidera confidence baserat pa faktiska varden
        row = _revalidate_confidence(row)
        rows.append(row)

    return rows


def _revalidate_confidence(row: CouponRow) -> CouponRow:
    """
    Omvaliderar confidence baserat pa vilka varden som faktiskt finns.

    Satter "incomplete" om kritiska falt saknas, "uncertain" om
    nagon data saknas men resten finns.
    """
    has_teams = bool(row.home_team and row.away_team)
    has_all_streck = all(v is not None for v in [row.streck_1, row.streck_x, row.streck_2])
    has_all_odds = all(v is not None for v in [row.odds_1, row.odds_x, row.odds_2])
    has_any_streck = any(v is not None for v in [row.streck_1, row.streck_x, row.streck_2])
    has_any_odds = any(v is not None for v in [row.odds_1, row.odds_x, row.odds_2])

    if not has_teams:
        row.confidence = "incomplete"
        if not row.notes:
            row.notes = "Lagnamn saknas"
    elif has_all_streck and has_all_odds:
        # Fullt komplett — behall rapporterad confidence
        if row.confidence not in ("ok", "uncertain"):
            row.confidence = "ok"
        # Validera att streck summerar till ~100% (±5%)
        streck_sum = row.streck_1 + row.streck_x + row.streck_2
        if abs(streck_sum - 100.0) > 5.0:
            row.confidence = "uncertain"
            row.notes = (
                row.notes + "; " if row.notes else ""
            ) + f"Streck summerar till {streck_sum:.1f}%, forvantade ~100%"
            logger.warning(
                "Streck for %s vs %s summerar till %.1f%% (forvantade ~100%%)",
                row.home_team,
                row.away_team,
                streck_sum,
            )
    elif not has_any_streck and not has_any_odds:
        row.confidence = "incomplete"
        if not row.notes:
            row.notes = "Varken streck eller odds kunde tolkas"
        logger.info(
            "Rad %s vs %s: varken streck eller odds kunde tolkas",
            row.home_team,
            row.away_team,
        )
    elif not has_all_streck or not has_all_odds:
        if row.confidence == "ok":
            row.confidence = "uncertain"
        if not row.notes:
            missing = []
            if not has_all_streck:
                missing.append("streck")
            if not has_all_odds:
                missing.append("odds")
            row.notes = f"Delvis saknade varden: {', '.join(missing)}"
        logger.info(
            "Rad %s vs %s: ofullstandig data (confidence=%s)",
            row.home_team,
            row.away_team,
            row.confidence,
        )

    # Validera streck-summa aven nar odds saknas men streck finns
    if has_all_streck and not has_all_odds:
        streck_sum = row.streck_1 + row.streck_x + row.streck_2
        if abs(streck_sum - 100.0) > 5.0:
            row.confidence = "uncertain"
            row.notes = (
                row.notes + "; " if row.notes else ""
            ) + f"Streck summerar till {streck_sum:.1f}%, forvantade ~100%"

    return row


def _build_result(rows: List[CouponRow], raw_response: str) -> CouponExtractionResult:
    """Bygger CouponExtractionResult fran en lista av rader."""
    total = len(rows)
    complete = sum(1 for r in rows if r.confidence == "ok")
    uncertain = sum(1 for r in rows if r.confidence == "uncertain")
    incomplete = sum(1 for r in rows if r.confidence == "incomplete")

    return CouponExtractionResult(
        rows=rows,
        total_rows=total,
        complete_rows=complete,
        uncertain_rows=uncertain,
        incomplete_rows=incomplete,
        raw_response=raw_response,
    )


# ---------------------------------------------------------------------------
# Hjalp: konvertera CouponRows till DataFrame
# ---------------------------------------------------------------------------

def coupon_rows_to_dataframe(rows: List[CouponRow]) -> "pd.DataFrame":
    """
    Konverterar CouponRow-lista till en pandas DataFrame lamplig for
    st.data_editor.

    Kolumner: HomeTeam, AwayTeam, Streck1, StreckX, Streck2,
              Odds1, OddsX, Odds2, Status, Notes
    """
    import pandas as pd

    records = []
    for row in rows:
        records.append({
            "HomeTeam": row.home_team,
            "AwayTeam": row.away_team,
            "Streck1": row.streck_1,
            "StreckX": row.streck_x,
            "Streck2": row.streck_2,
            "Odds1": row.odds_1,
            "OddsX": row.odds_x,
            "Odds2": row.odds_2,
            "Status": row.confidence,
            "Notes": row.notes,
        })

    return pd.DataFrame(records)


def dataframe_to_coupon_rows(df: "pd.DataFrame") -> List[CouponRow]:
    """
    Konverterar en DataFrame (fran st.data_editor) tillbaka till CouponRow-lista.
    """
    import pandas as pd

    rows: List[CouponRow] = []
    for _, r in df.iterrows():
        home = str(r.get("HomeTeam", "")).strip()
        away = str(r.get("AwayTeam", "")).strip()
        if not home and not away:
            continue

        row = CouponRow(
            home_team=home,
            away_team=away,
            streck_1=_safe_float(r.get("Streck1")),
            streck_x=_safe_float(r.get("StreckX")),
            streck_2=_safe_float(r.get("Streck2")),
            odds_1=_safe_odds(r.get("Odds1")),
            odds_x=_safe_odds(r.get("OddsX")),
            odds_2=_safe_odds(r.get("Odds2")),
            confidence=str(r.get("Status", "ok")).strip().lower(),
            notes=str(r.get("Notes", "")).strip(),
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Hjalp: bygg fixtures, odds, streck fran bekraftade rader
# ---------------------------------------------------------------------------

def confirmed_rows_to_matchday_data(
    rows: List[CouponRow],
) -> Tuple[
    List[Any],           # fixtures: List[MatchdayFixture]
    Dict[str, List],     # odds_by_key
    Dict[str, Dict],     # streck_by_key
    List[str],           # rows_with_image_odds (match keys)
    List[str],           # rows_missing_odds (match keys)
]:
    """
    Konverterar bekraftade CouponRows till matchday-datastrukturer
    kompatibla med match_matchday_data().

    Returnerar fixtures, odds_by_key, streck_by_key,
    lista av nycklar med bildodds, lista av nycklar utan odds.
    """
    from matchday_import import MatchdayFixture, _make_key
    from odds_tool import OddsEntry

    fixtures: List[MatchdayFixture] = []
    odds_by_key: Dict[str, List[OddsEntry]] = {}
    streck_by_key: Dict[str, Dict[str, float]] = {}
    rows_with_image_odds: List[str] = []
    rows_missing_odds: List[str] = []

    for row in rows:
        if not row.home_team or not row.away_team:
            continue

        key = _make_key(row.home_team, row.away_team)

        fixtures.append(MatchdayFixture(
            home_team=row.home_team,
            away_team=row.away_team,
            match_key=key,
        ))

        # Odds fran bilden
        if row.odds_1 is not None and row.odds_x is not None and row.odds_2 is not None:
            if row.odds_1 > 1.0 and row.odds_x > 1.0 and row.odds_2 > 1.0:
                odds_by_key[key] = [OddsEntry(
                    bookmaker="Kupongbild",
                    home=row.odds_1,
                    draw=row.odds_x,
                    away=row.odds_2,
                )]
                rows_with_image_odds.append(key)
            else:
                rows_missing_odds.append(key)
        else:
            rows_missing_odds.append(key)

        # Streck fran bilden
        if row.streck_1 is not None and row.streck_x is not None and row.streck_2 is not None:
            total = row.streck_1 + row.streck_x + row.streck_2
            if total > 0:
                streck_by_key[key] = {
                    "1": row.streck_1,
                    "X": row.streck_x,
                    "2": row.streck_2,
                }

    return fixtures, odds_by_key, streck_by_key, rows_with_image_odds, rows_missing_odds
