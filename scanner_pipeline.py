"""
scanner_pipeline.py -- Robust kupongbild-scanner med pipeline-arkitektur.

Pipeline-steg:
  1. Bildforbehandling (kontrast, skalning, binarisering, deskew)
  2. Forbattrad AI-prompt for radvis extraktion
  3. Faltvis parsning och validering per rad
  4. Canonical team mapping (fuzzy-match mot kanda lag)
  5. Confidence/uncertainty per rad och falt
  6. Strukturerat resultat for UI

Anvandning:
    from scanner_pipeline import run_scanner_pipeline, ScannerResult

    result = run_scanner_pipeline(image_bytes, "coupon.png")
    for row in result.rows:
        print(row.home_team, row.confidence_score, row.row_status)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataklasser for pipeline-output
# ---------------------------------------------------------------------------

@dataclass
class FieldConfidence:
    """Confidence for ett enskilt falt."""
    value: Any = None
    raw_text: str = ""
    confidence: float = 1.0  # 0.0-1.0
    issue: str = ""


@dataclass
class ScannedRow:
    """En rad fran scannerpipelinen med faltvis confidence."""
    home_team: str = ""
    away_team: str = ""
    home_team_raw: str = ""  # Ra OCR-text fore mapping
    away_team_raw: str = ""
    streck_1: Optional[float] = None
    streck_x: Optional[float] = None
    streck_2: Optional[float] = None
    odds_1: Optional[float] = None
    odds_x: Optional[float] = None
    odds_2: Optional[float] = None

    # Faltvis confidence
    home_team_confidence: float = 1.0
    away_team_confidence: float = 1.0
    streck_confidence: float = 1.0
    odds_confidence: float = 1.0

    # Radniva
    confidence_score: float = 1.0  # 0.0-1.0 aggregerad
    row_status: str = "ok"  # "ok", "uncertain", "failed"
    issues: List[str] = field(default_factory=list)
    notes: str = ""

    # Mapping-info
    home_team_mapped: bool = False
    away_team_mapped: bool = False


@dataclass
class ScannerResult:
    """Resultat fran hela scannerpipelinen."""
    rows: List[ScannedRow]
    total_rows: int = 0
    ok_rows: int = 0
    uncertain_rows: int = 0
    failed_rows: int = 0
    raw_response: str = ""
    error: Optional[str] = None
    preprocessing_applied: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Steg 1: Bildforbehandling
# ---------------------------------------------------------------------------

def preprocess_image(image_bytes: bytes) -> Tuple[bytes, List[str]]:
    """
    Forbehandlar kupongbild for battre OCR-resultat.

    Steg:
    - Konvertera till RGB
    - Skala upp om bilden ar for liten
    - Forbattra kontrast (autocontrast)
    - Latt sharpen for tydligare text
    - Kropa bort marginaler om mojligt

    Returnerar (bearbetade bytes som PNG, lista av tillampade steg).
    """
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    except ImportError:
        logger.warning("Pillow inte tillgangligt, hoppar over bildforbehandling")
        return image_bytes, ["skip: Pillow ej installerat"]

    steps_applied: List[str] = []

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        logger.warning("Kunde inte oppna bild for forbehandling: %s", e)
        return image_bytes, [f"skip: kunde inte oppna bild ({e})"]

    # Konvertera till RGB om nodvandigt
    if img.mode != "RGB":
        img = img.convert("RGB")
        steps_applied.append("convert_rgb")

    # Skala upp om bilden ar for liten (under 1200px bred)
    width, height = img.size
    min_width = 1200
    if width < min_width:
        scale = min_width / width
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, Image.LANCZOS)
        steps_applied.append(f"upscale_{scale:.1f}x")

    # Autokontrast for att normalisera ljusniva
    img = ImageOps.autocontrast(img, cutoff=1)
    steps_applied.append("autocontrast")

    # Forbattra kontrast ytterligare
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    steps_applied.append("contrast_1.3")

    # Latt sharpening for tydligare text
    img = img.filter(ImageFilter.SHARPEN)
    steps_applied.append("sharpen")

    # Kropa bort overflodiga marginaler (hitta bounding box av icke-vit)
    img_cropped = _smart_crop(img)
    if img_cropped is not None:
        img = img_cropped
        steps_applied.append("smart_crop")

    # Exportera som PNG
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), steps_applied


def _smart_crop(img: Any) -> Any:
    """
    Kropa bort overflodiga vita/ljusa marginaler runt kuponginnehallet.
    Returnerar beskuren bild eller None om ingen kropping behovs.
    """
    try:
        from PIL import ImageOps
    except ImportError:
        return None

    try:
        # Konvertera till gra for att hitta innehallskanter
        gray = img.convert("L")
        # Invertera sa att text/innehall ar ljust
        inverted = ImageOps.invert(gray)
        # Hitta bounding box av icke-svart (dvs originalinnehall)
        bbox = inverted.getbbox()
        if bbox is None:
            return None

        # Lagg till liten marginal
        margin = 10
        x0 = max(0, bbox[0] - margin)
        y0 = max(0, bbox[1] - margin)
        x1 = min(img.width, bbox[2] + margin)
        y1 = min(img.height, bbox[3] + margin)

        # Kropa bara om det finns meningsfull marginal att ta bort
        orig_area = img.width * img.height
        crop_area = (x1 - x0) * (y1 - y0)
        if crop_area < orig_area * 0.90:
            return img.crop((x0, y0, x1, y1))
    except Exception as e:
        logger.debug("Smart crop misslyckades: %s", e)

    return None


# ---------------------------------------------------------------------------
# Steg 2: Forbattrad AI-prompt for radvis extraktion
# ---------------------------------------------------------------------------

_ENHANCED_EXTRACTION_PROMPT = """Du ar en expert pa att lasa svenska kupongbilder (tipsbilder/stryktipset/europatipset).

UPPGIFT:
Lasa bilden rad for rad. Varje rad ar EN match. Extrahera foljande falt SEPARAT for varje rad:

FALT ATT EXTRAHERA PER RAD:
1. HomeTeam - Hemmalaget (vanster lag)
2. AwayTeam - Bortalaget (hoger lag)
3. Streck1 - Streckprocent for hemmavinst (1). Tal 0-100.
4. StreckX - Streckprocent for oavgjort (X). Tal 0-100.
5. Streck2 - Streckprocent for bortavinst (2). Tal 0-100.
6. Odds1 - Decimalodds for hemmavinst. Tal > 1.0 (t.ex. 2.10)
7. OddsX - Decimalodds for oavgjort. Tal > 1.0 (t.ex. 3.40)
8. Odds2 - Decimalodds for bortavinst. Tal > 1.0 (t.ex. 3.60)
9. row_confidence - Hur saker du ar pa hela raden: "high", "medium", "low"
10. field_issues - Lista av falt som var svarlasta (t.ex. ["Odds1", "AwayTeam"])

REGLER:
- Returnera ENBART en JSON-array. Ingen markdown, inga code fences.
- En JSON-objekt per matchrad i bilden.
- Lagnamen ska vara SA EXAKTA som mojligt fran bilden. Gissa INTE.
- Streckprocent: heltal eller decimaltal 0-100. Summa ~100 per rad.
- Odds: decimaltal > 1.0.
- Om ett falt ar olasbart, satt det till null.
- Om text ar pixlad/suddig, satt row_confidence till "low" och lista faltet i field_issues.
- Blanda INTE ihop streckprocent och odds. Streck ar procent (t.ex. 45), odds ar decimaltal (t.ex. 2.10).
- Procentsiffror ar ALDRIG over 100 per falt.
- Oddsvarden har ALLTID en decimalpunkt och ar normalt mellan 1.01 och 50.0.

LAYOUT-TIPS:
- Kupongen har vanligtvis kolumner: Rad/Nr | Match/Lag | 1 | X | 2 (streck) | 1 | X | 2 (odds)
- Hemmalaget star vanligtvis fore bortalaget, separerat med " - " eller pa tva rader.
- Streckkolumnerna (procent) och oddskolumnerna (decimaltal) kan ligga bredvid varandra.
- Skilj pa streck (heltal 0-100) och odds (decimaltal > 1.0) baserat pa talformat.

EXEMPEL:
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
    "row_confidence": "high",
    "field_issues": []
  }
]"""


# ---------------------------------------------------------------------------
# Steg 3: Faltvis parsning och validering
# ---------------------------------------------------------------------------

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


def _safe_streck(value: Any) -> Optional[float]:
    """Konverterar till streckprocent (0-100) om mojligt, annars None."""
    f = _safe_float(value)
    if f is not None and 0.0 <= f <= 100.0:
        return f
    return None


def parse_row_fields(item: Dict[str, Any]) -> ScannedRow:
    """
    Parsar och validerar alla falt fran en JSON-rad.
    Satter faltvis confidence baserat pa parsningsresultat.
    """
    row = ScannedRow()

    # Lagnamn
    home_raw = str(item.get("HomeTeam", "")).strip()
    away_raw = str(item.get("AwayTeam", "")).strip()
    row.home_team_raw = home_raw
    row.away_team_raw = away_raw
    row.home_team = home_raw
    row.away_team = away_raw

    if not home_raw:
        row.home_team_confidence = 0.0
        row.issues.append("Hemmalag saknas")
    if not away_raw:
        row.away_team_confidence = 0.0
        row.issues.append("Bortalag saknas")

    # Streck
    row.streck_1 = _safe_streck(item.get("Streck1"))
    row.streck_x = _safe_streck(item.get("StreckX"))
    row.streck_2 = _safe_streck(item.get("Streck2"))

    streck_values = [row.streck_1, row.streck_x, row.streck_2]
    streck_present = [v for v in streck_values if v is not None]

    if len(streck_present) == 3:
        streck_sum = sum(streck_present)
        if abs(streck_sum - 100.0) <= 5.0:
            row.streck_confidence = 1.0
        elif abs(streck_sum - 100.0) <= 10.0:
            row.streck_confidence = 0.6
            row.issues.append(f"Streck summerar till {streck_sum:.1f}%")
        else:
            row.streck_confidence = 0.3
            row.issues.append(f"Streck summerar till {streck_sum:.1f}% (forvantade ~100%)")
    elif len(streck_present) > 0:
        row.streck_confidence = 0.4
        row.issues.append("Ofullstandiga streckvarden")
    else:
        row.streck_confidence = 0.0

    # Odds
    row.odds_1 = _safe_odds(item.get("Odds1"))
    row.odds_x = _safe_odds(item.get("OddsX"))
    row.odds_2 = _safe_odds(item.get("Odds2"))

    odds_values = [row.odds_1, row.odds_x, row.odds_2]
    odds_present = [v for v in odds_values if v is not None]

    if len(odds_present) == 3:
        row.odds_confidence = _validate_odds_triplet(
            row.odds_1, row.odds_x, row.odds_2, row.issues
        )
    elif len(odds_present) > 0:
        row.odds_confidence = 0.4
        row.issues.append("Ofullstandiga oddsvarden")
    else:
        row.odds_confidence = 0.0

    # Kolla for forvirrning mellan streck och odds
    _check_streck_odds_confusion(row)

    # API-rapporterad confidence
    api_confidence = str(item.get("row_confidence", "")).strip().lower()
    api_field_issues = item.get("field_issues", [])
    if isinstance(api_field_issues, list) and api_field_issues:
        row.issues.append(f"AI-rapporterade problem: {', '.join(str(f) for f in api_field_issues)}")

    # Justera baserat pa API-confidence
    api_multiplier = {"high": 1.0, "medium": 0.8, "low": 0.5}.get(api_confidence, 0.9)

    # Berakna aggregerad confidence
    row.confidence_score = _compute_row_confidence(row, api_multiplier)
    row.row_status = _determine_row_status(row)

    return row


def _validate_odds_triplet(
    odds_1: float, odds_x: float, odds_2: float, issues: List[str]
) -> float:
    """
    Validerar att tre odds ar rimliga som ett trippelt.
    Returnerar confidence 0.0-1.0.
    """
    # Berakna implicit sannolikhet (overround)
    try:
        implied_sum = (1.0 / odds_1) + (1.0 / odds_x) + (1.0 / odds_2)
    except ZeroDivisionError:
        issues.append("Odds innehaller noll")
        return 0.2

    # Rimlighetskontroll: enskilda odds bor vara inom normalt intervall
    for label, val in [("Odds1", odds_1), ("OddsX", odds_x), ("Odds2", odds_2)]:
        if val > 100.0:
            issues.append(f"{label}={val:.2f} ar orimligt hogt")
            return 0.2
        elif val > 50.0:
            issues.append(f"{label}={val:.2f} ar ovanligt hogt")
            return 0.5

    # Overround bor vara ~1.0-1.3 (0-30% marginal)
    if 0.9 <= implied_sum <= 1.4:
        return 1.0
    elif 0.8 <= implied_sum <= 1.6:
        issues.append(f"Overround {implied_sum:.2f} ar ovanligt")
        return 0.6
    else:
        issues.append(f"Overround {implied_sum:.2f} ar orimligt")
        return 0.3


def _check_streck_odds_confusion(row: ScannedRow) -> None:
    """
    Detekterar om streck- och oddsvarden kan ha blandats ihop.
    T.ex. om streckvarden ser ut som odds (1.xx-10.xx) eller
    oddsvarden ser ut som procent (20-60).
    """
    # Kolla om streckvarden ser ut som odds (alla < 15 och > 1)
    streck_vals = [v for v in [row.streck_1, row.streck_x, row.streck_2] if v is not None]
    if len(streck_vals) == 3:
        if all(1.0 < v < 15.0 for v in streck_vals):
            # Kan vara odds istallet for procent
            implied = sum(1.0 / v for v in streck_vals)
            if 0.85 <= implied <= 1.5:
                row.issues.append(
                    "Streckvarden kan vara forvirrda med odds "
                    f"({streck_vals[0]:.2f}/{streck_vals[1]:.2f}/{streck_vals[2]:.2f})"
                )
                row.streck_confidence = min(row.streck_confidence, 0.3)

    # Kolla om oddsvarden ser ut som procent (alla 10-60)
    odds_vals = [v for v in [row.odds_1, row.odds_x, row.odds_2] if v is not None]
    if len(odds_vals) == 3:
        if all(10.0 < v < 70.0 for v in odds_vals):
            total = sum(odds_vals)
            if 80.0 <= total <= 120.0:
                row.issues.append(
                    "Oddsvarden kan vara forvirrda med streckprocent "
                    f"({odds_vals[0]:.1f}/{odds_vals[1]:.1f}/{odds_vals[2]:.1f})"
                )
                row.odds_confidence = min(row.odds_confidence, 0.3)


def _compute_row_confidence(row: ScannedRow, api_multiplier: float) -> float:
    """Beraknar aggregerad confidence for en rad."""
    # Vikter for olika faltgrupper
    team_weight = 0.35
    streck_weight = 0.25
    odds_weight = 0.25
    api_weight = 0.15

    team_conf = min(row.home_team_confidence, row.away_team_confidence)

    score = (
        team_conf * team_weight
        + row.streck_confidence * streck_weight
        + row.odds_confidence * odds_weight
        + api_multiplier * api_weight
    )

    # Straffa rader med manga issues
    if len(row.issues) > 3:
        score *= 0.8
    if len(row.issues) > 5:
        score *= 0.7

    return max(0.0, min(1.0, score))


def _determine_row_status(row: ScannedRow) -> str:
    """Bestammer radstatus baserat pa confidence."""
    if row.confidence_score >= 0.7:
        return "ok"
    elif row.confidence_score >= 0.4:
        return "uncertain"
    else:
        return "failed"


# ---------------------------------------------------------------------------
# Steg 4: Canonical team mapping
# ---------------------------------------------------------------------------

# Utokad lista av kanda lag for scanner-specifik fuzzy matching
SCANNER_TEAM_ALIASES: Dict[str, str] = {
    # Vanliga OCR-felaktigheter och forenklingar
    "Man Utd": "Manchester United",
    "Man United": "Manchester United",
    "Man U": "Manchester United",
    "ManUtd": "Manchester United",
    "Man City": "Manchester City",
    "ManCity": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "Nottm Forest": "Nottingham Forest",
    "Nott Forest": "Nottingham Forest",
    "Nottingham F": "Nottingham Forest",
    "N Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Sheff Utd": "Sheffield United",
    "Sheffield Utd": "Sheffield United",
    "Sheff United": "Sheffield United",
    "Sheff Wed": "Sheffield Wednesday",
    "Sheffield Wed": "Sheffield Wednesday",
    "Sheffield Weds": "Sheffield Wednesday",
    "Spurs": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "Newcastle": "Newcastle United",
    "Newcastle Utd": "Newcastle United",
    "Brighton": "Brighton & Hove Albion",
    "Brighton Hove": "Brighton & Hove Albion",
    "West Ham": "West Ham United",
    "West Brom": "West Bromwich Albion",
    "West Bromwich": "West Bromwich Albion",
    "QPR": "Queens Park Rangers",
    "Birmingham": "Birmingham City",
    "Blackburn": "Blackburn Rovers",
    "Bristol C": "Bristol City",
    "Bristol City": "Bristol City",
    "Cardiff": "Cardiff City",
    "Huddersfield": "Huddersfield Town",
    "Leicester": "Leicester City",
    "Norwich": "Norwich City",
    "Stoke": "Stoke City",
    "Swansea": "Swansea City",
    "Preston": "Preston North End",
    "Charlton": "Charlton Athletic",
    "Oxford": "Oxford United",
    "Stockport": "Stockport County",
    "Wigan": "Wigan Athletic",
    "Luton": "Luton Town",
    "Plymouth": "Plymouth Argyle",
    "Derby": "Derby County",
    "Leeds": "Leeds United",
    "Ipswich": "Ipswich Town",
    "Rotherham": "Rotherham United",
    "Peterborough": "Peterborough United",
    "Peterboro": "Peterborough United",
    "Cambridge": "Cambridge United",
    "Cambridge Utd": "Cambridge United",
    # Vanliga OCR-missuppfattningar med specialtecken
    "Brighton &": "Brighton & Hove Albion",
    "Brighton&Hove": "Brighton & Hove Albion",
}


def map_team_canonical(
    raw_name: str,
    canonical_set: Optional[Set[str]] = None,
) -> Tuple[str, float, bool]:
    """
    Matchar ett OCR-last lagnamn mot kanda kanoniska lag.

    Returnerar (mappat namn, confidence 0.0-1.0, om mapping gjordes).
    """
    if not raw_name or not raw_name.strip():
        return raw_name, 0.0, False

    name = _normalize_text(raw_name)

    # 1) Exakt match mot scanner-alias
    if name in SCANNER_TEAM_ALIASES:
        return SCANNER_TEAM_ALIASES[name], 1.0, True

    # Case-insensitive alias
    name_lower = name.lower()
    for alias, canonical in SCANNER_TEAM_ALIASES.items():
        if alias.lower() == name_lower:
            return canonical, 1.0, True

    # 2) Forsoker anvanda utils.normalize_team_name om canonical_set finns
    if canonical_set:
        from utils import normalize_team_name, set_canonical_teams, get_canonical_teams

        # Temporart satt canonical teams om inte redan satta
        existing = get_canonical_teams()
        if not existing and canonical_set:
            set_canonical_teams(canonical_set)

        normalized = normalize_team_name(name)
        if normalized != name:
            return normalized, 0.9, True

    # 3) Fuzzy match mot scanner-alias (for OCR-fel)
    best_match, best_score = _fuzzy_match_team(name, canonical_set)
    if best_match and best_score >= 0.75:
        confidence = best_score
        return best_match, confidence, True

    # Ingen match hittad
    return name, 0.5, False


def _normalize_text(s: str) -> str:
    """Grundlaggande textnormalisering for lagnamn."""
    # Normalisera mellanslag och bindestreck
    s = re.sub(r"[–—−]", "-", s)
    s = re.sub(r"[\.]", "", s)
    s = s.replace("'", "'").replace("\u00b4", "'")
    s = re.sub(r"\s+", " ", s.strip())
    return s


def _fuzzy_match_team(
    name: str,
    canonical_set: Optional[Set[str]] = None,
) -> Tuple[Optional[str], float]:
    """
    Fuzzy-matchar lagnamn mot kanda lag.
    Returnerar (basta match, score) eller (None, 0.0).
    """
    from difflib import SequenceMatcher

    candidates: Dict[str, str] = {}

    # Lagg till scanner-alias
    for alias, canonical in SCANNER_TEAM_ALIASES.items():
        candidates[alias.lower()] = canonical

    # Lagg till canonical set
    if canonical_set:
        for c in canonical_set:
            candidates[c.lower()] = c

    target = name.lower()
    best_name: Optional[str] = None
    best_score = 0.0

    for key, value in candidates.items():
        score = SequenceMatcher(None, target, key).ratio()
        if score > best_score:
            best_score = score
            best_name = value

    return best_name, best_score


def apply_team_mapping(
    rows: List[ScannedRow],
    canonical_set: Optional[Set[str]] = None,
) -> List[ScannedRow]:
    """
    Applicerar canonical team mapping pa alla rader.
    Uppdaterar lagnamn, confidence och mapping-flaggor.
    """
    for row in rows:
        if row.home_team:
            mapped, conf, was_mapped = map_team_canonical(row.home_team, canonical_set)
            row.home_team = mapped
            row.home_team_mapped = was_mapped
            row.home_team_confidence = conf

        if row.away_team:
            mapped, conf, was_mapped = map_team_canonical(row.away_team, canonical_set)
            row.away_team = mapped
            row.away_team_mapped = was_mapped
            row.away_team_confidence = conf

        # Rakna om aggregerad confidence efter mapping
        row.confidence_score = _compute_row_confidence(row, 1.0)
        row.row_status = _determine_row_status(row)

    return rows


# ---------------------------------------------------------------------------
# Steg 5: Radvalidering
# ---------------------------------------------------------------------------

def validate_rows(rows: List[ScannedRow]) -> List[ScannedRow]:
    """
    Korsvaliderar alla rader med rimlighetskontroller.
    """
    for row in rows:
        _validate_single_row(row)
        # Uppdatera status efter validering
        row.row_status = _determine_row_status(row)

    return rows


def _validate_single_row(row: ScannedRow) -> None:
    """
    Validerar en enskild rad med regler:
    - Bada lag maste finnas
    - Strecksumma ~100%
    - Oddsvarden rimliga
    - Ingen dubblering
    """
    # Regel 1: Bada lag maste finnas
    if not row.home_team or not row.away_team:
        row.confidence_score *= 0.3
        if not row.home_team and not row.away_team:
            row.row_status = "failed"
            row.issues.append("Bade hemma- och bortalag saknas")
            return

    # Regel 2: Lag bor inte vara identiska
    if row.home_team and row.away_team:
        if row.home_team.lower() == row.away_team.lower():
            row.issues.append("Hemma- och bortalag ar identiska")
            row.confidence_score *= 0.3

    # Regel 3: Validera strecksumma
    streck_vals = [row.streck_1, row.streck_x, row.streck_2]
    if all(v is not None for v in streck_vals):
        streck_sum = sum(streck_vals)
        if abs(streck_sum - 100.0) > 10.0:
            row.issues.append(f"Strecksumma {streck_sum:.1f}% avviker kraftigt fran 100%")
            row.confidence_score *= 0.5
        elif abs(streck_sum - 100.0) > 5.0:
            row.issues.append(f"Strecksumma {streck_sum:.1f}% avviker nagot fran 100%")
            row.confidence_score *= 0.8

    # Regel 4: Validera enskilda oddsvarden
    for label, val in [("Odds1", row.odds_1), ("OddsX", row.odds_x), ("Odds2", row.odds_2)]:
        if val is not None:
            if val > 100.0:
                row.issues.append(f"{label}={val:.2f} ar orimligt hogt")
                row.confidence_score *= 0.5
            elif val > 50.0:
                row.issues.append(f"{label}={val:.2f} ar ovanligt hogt")
                row.confidence_score *= 0.8

    # Regel 5: Kolla overround om alla odds finns
    odds_vals = [row.odds_1, row.odds_x, row.odds_2]
    if all(v is not None for v in odds_vals):
        try:
            implied = sum(1.0 / v for v in odds_vals)
            if implied < 0.85 or implied > 1.5:
                row.issues.append(f"Overround {implied:.2f} ar utanfor normalt intervall")
                row.confidence_score *= 0.6
        except ZeroDivisionError:
            pass

    # Regel 6: Streckvarden bor inte vara exakt 0 alla tre
    if all(v == 0 for v in streck_vals if v is not None) and any(v is not None for v in streck_vals):
        row.issues.append("Alla streckvarden ar 0")
        row.streck_confidence = 0.1
        row.confidence_score *= 0.5

    # Klamp confidence
    row.confidence_score = max(0.0, min(1.0, row.confidence_score))


# ---------------------------------------------------------------------------
# Steg 6: Huvudpipeline
# ---------------------------------------------------------------------------

def run_scanner_pipeline(
    image_bytes: bytes,
    filename: str = "coupon.png",
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    canonical_teams: Optional[Set[str]] = None,
) -> ScannerResult:
    """
    Kor hela scannerpipelinen:
    1. Bildforbehandling
    2. AI-extraktion med forbattrad prompt
    3. Faltvis parsning
    4. Canonical team mapping
    5. Validering
    6. Bygger strukturerat resultat

    Parametrar
    ----------
    image_bytes : bytes
        Rabildbytes (PNG, JPG, WEBP).
    filename : str
        Filnamn for MIME-typ-detektion.
    api_key : str, optional
        OpenAI API-nyckel.
    model : str
        OpenAI-modell. Standard: gpt-4o-mini.
    canonical_teams : set, optional
        Mangd av kanda kanoniska lagnamn for mapping.

    Returnerar
    ----------
    ScannerResult med tolkade rader, confidence och statistik.
    """
    import os

    # Validera format
    from coupon_image_parser import SUPPORTED_EXTENSIONS
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return ScannerResult(
            rows=[],
            error=f"Bildformatet stods inte: {ext}. "
                  f"Stodda format: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # Hamta API-nyckel
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass
    if not api_key:
        return ScannerResult(
            rows=[],
            error="Ingen OpenAI API-nyckel tillganglig. Satt OPENAI_API_KEY.",
        )

    # Steg 1: Bildforbehandling
    processed_bytes, preprocess_steps = preprocess_image(image_bytes)
    logger.info("Bildforbehandling klar: %s", preprocess_steps)

    # Steg 2: AI-extraktion
    raw_text, ai_error = _call_openai_vision(processed_bytes, filename, api_key, model)
    if ai_error:
        return ScannerResult(
            rows=[],
            error=ai_error,
            preprocessing_applied=preprocess_steps,
        )

    # Steg 3: Parsa JSON-svar till rader
    rows, parse_error = _parse_ai_response(raw_text)
    if parse_error:
        return ScannerResult(
            rows=[],
            raw_response=raw_text,
            error=parse_error,
            preprocessing_applied=preprocess_steps,
        )

    # Steg 4: Canonical team mapping
    rows = apply_team_mapping(rows, canonical_teams)

    # Steg 5: Validering
    rows = validate_rows(rows)

    # Steg 6: Bygg resultat
    return _build_scanner_result(rows, raw_text, preprocess_steps)


def _call_openai_vision(
    image_bytes: bytes,
    filename: str,
    api_key: str,
    model: str,
) -> Tuple[str, Optional[str]]:
    """
    Anropar OpenAI Vision API med forbattrad prompt.
    Returnerar (raw_text, error_message).
    """
    try:
        from openai import OpenAI
    except ImportError:
        return "", "OpenAI-paketet ar inte installerat. Kor: pip install openai"

    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    ext = Path(filename).suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg", ".webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/png")

    client = OpenAI(api_key=api_key)
    max_retries = 2

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _ENHANCED_EXTRACTION_PROMPT},
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
            return raw_text, None
        except Exception as e:
            logger.warning(
                "OpenAI Vision API forsok %d/%d misslyckades: %s",
                attempt, max_retries, e,
            )
            if attempt == max_retries:
                return "", f"OpenAI API-fel efter {max_retries} forsok: {e}"

    return "", "Okant fel i OpenAI-anrop"


def _parse_ai_response(raw_text: str) -> Tuple[List[ScannedRow], Optional[str]]:
    """
    Parsar AI-svaret fran JSON till ScannedRow-lista.
    Returnerar (rows, error_message).
    """
    json_text = raw_text

    # Extrahera JSON fran eventuell markdown
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw_text)
    if json_match:
        json_text = json_match.group(1).strip()

    # Hitta array-start
    if not json_text.startswith("["):
        bracket_start = json_text.find("[")
        if bracket_start >= 0:
            json_text = json_text[bracket_start:]

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as e:
        return [], f"Kunde inte tolka API-svaret som JSON: {e}"

    if not isinstance(parsed, list):
        return [], "API-svaret ar inte en lista."

    rows: List[ScannedRow] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue

        home = str(item.get("HomeTeam", "")).strip()
        away = str(item.get("AwayTeam", "")).strip()
        if not home and not away:
            continue

        row = parse_row_fields(item)
        rows.append(row)

    return rows, None


def _build_scanner_result(
    rows: List[ScannedRow],
    raw_response: str,
    preprocess_steps: List[str],
) -> ScannerResult:
    """Bygger ScannerResult fran validerade rader."""
    ok = sum(1 for r in rows if r.row_status == "ok")
    uncertain = sum(1 for r in rows if r.row_status == "uncertain")
    failed = sum(1 for r in rows if r.row_status == "failed")

    return ScannerResult(
        rows=rows,
        total_rows=len(rows),
        ok_rows=ok,
        uncertain_rows=uncertain,
        failed_rows=failed,
        raw_response=raw_response,
        preprocessing_applied=preprocess_steps,
    )


# ---------------------------------------------------------------------------
# Konvertering: ScannedRow <-> CouponRow (for bakatkompabilitet)
# ---------------------------------------------------------------------------

def scanned_rows_to_coupon_rows(rows: List[ScannedRow]) -> List:
    """
    Konverterar ScannedRow-lista till CouponRow-lista for bakatkompabilitet
    med befintlig kod (coupon_image_parser.CouponRow).
    """
    from coupon_image_parser import CouponRow

    result = []
    for row in rows:
        # Mappa row_status till confidence-strang
        confidence_map = {"ok": "ok", "uncertain": "uncertain", "failed": "incomplete"}
        confidence = confidence_map.get(row.row_status, "uncertain")

        coupon_row = CouponRow(
            home_team=row.home_team,
            away_team=row.away_team,
            streck_1=row.streck_1,
            streck_x=row.streck_x,
            streck_2=row.streck_2,
            odds_1=row.odds_1,
            odds_x=row.odds_x,
            odds_2=row.odds_2,
            confidence=confidence,
            notes=row.notes if row.notes else "; ".join(row.issues) if row.issues else "",
        )
        result.append(coupon_row)

    return result


def scanner_result_to_extraction_result(scanner_result: ScannerResult):
    """
    Konverterar ScannerResult till CouponExtractionResult for bakatkompabilitet.
    """
    from coupon_image_parser import CouponExtractionResult

    coupon_rows = scanned_rows_to_coupon_rows(scanner_result.rows)

    return CouponExtractionResult(
        rows=coupon_rows,
        total_rows=scanner_result.total_rows,
        complete_rows=scanner_result.ok_rows,
        uncertain_rows=scanner_result.uncertain_rows,
        incomplete_rows=scanner_result.failed_rows,
        raw_response=scanner_result.raw_response,
        error=scanner_result.error,
    )
