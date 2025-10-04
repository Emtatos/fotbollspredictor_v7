# utils.py — robust namnnormlisering för E0–E2
from __future__ import annotations
from typing import Iterable, Optional, Set, Dict
from difflib import SequenceMatcher
import re

# =========================
#  Global kanonisk uppsättning
# =========================
_CANONICAL_TEAMS: Set[str] = set()

def set_canonical_teams(names: Iterable[str]) -> None:
    """
    Mata in alla lagnamn som finns i rå-DF (HomeTeam/AwayTeam).
    Detta gör att vi kan fuzzy-matcha mot EXAKT de namn som
    faktiskt finns i dina CSV:er för E0–E2 (oavsett säsong/division).
    """
    global _CANONICAL_TEAMS
    cleaned = { _norm_space(n) for n in names if isinstance(n, str) and n.strip() }
    _CANONICAL_TEAMS = cleaned

def get_canonical_teams() -> Set[str]:
    return set(_CANONICAL_TEAMS)

# =========================
#  Hjälp-funktioner
# =========================
_SPACE_RE = re.compile(r"\s+")
_DASH_RE  = re.compile(r"[–—−]")  # en/em/minus → '-'
_DOT_RE   = re.compile(r"[\.]")

def _norm_space(s: str) -> str:
    s = _DASH_RE.sub("-", s)
    s = _DOT_RE.sub("", s)
    s = s.replace("’", "'").replace("´", "'")
    s = _SPACE_RE.sub(" ", s.strip())
    return s

def _lower_key(s: str) -> str:
    return _norm_space(s).lower()

def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

# =========================
#  Manuella alias (vanliga kortformer)
#  OBS: så liten som möjligt – fuzzy tar resten
# =========================
TEAM_ALIASES: Dict[str, str] = {
    # Premier League (exempel + vanliga)
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "Nottm Forest": "Nottingham Forest",
    "Nottingham F": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers",
    "Sheff Utd": "Sheffield United",
    "Sheffield Utd": "Sheffield United",
    "Sheff Wed": "Sheffield Wednesday",
    "Sheffield Wed": "Sheffield Wednesday",
    "Sheffield Weds": "Sheffield Wednesday",
    "Spurs": "Tottenham Hotspur",
    "Newcastle": "Newcastle United",
    "Brighton": "Brighton & Hove Albion",
    "West Ham": "West Ham United",
    "West Brom": "West Bromwich Albion",
    "West Bromwich": "West Bromwich Albion",

    # Championship / L1 vanliga
    "QPR": "Queens Park Rangers",
    "Birmingham": "Birmingham City",
    "Blackburn": "Blackburn Rovers",
    "Bristol C": "Bristol City",
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
    "MK Dons": "MK Dons",
    "Millwall": "Millwall",
    "Derby": "Derby County",
    "Portsmouth": "Portsmouth",
    "Middlesbrough": "Middlesbrough",
    "Sunderland": "Sunderland",
    "Watford": "Watford",
    "Leeds": "Leeds United",
    "Ipswich": "Ipswich Town",
    "Rotherham": "Rotherham United",
}

# Token-utbyten vi provar innan fuzzy (billiga och generella)
TOKEN_REPLACE: Dict[str, str] = {
    "utd": "united",
    "man": "manchester",    # bara om det hjälper fuzzy
    "fc": "",               # ofta redundant i csv
    "afc": "",              # ofta redundant i csv
    "c": "city",            # hanterar " C" -> " City" i vissa källor
}

def _cheap_normal_forms(s: str) -> Set[str]:
    """
    Bygger några billiga varianter (utan att bli aggressiv).
    """
    out = set()
    base = _norm_space(s)
    out.add(base)

    low = base.lower()
    tokens = low.split()

    # 1) Ta bort suffix 'FC'/'AFC'
    if tokens and tokens[-1] in ("fc", "afc"):
        out.add(_norm_space(" ".join(tokens[:-1])))

    # 2) Byt ut tokens enligt TOKEN_REPLACE (försiktigt)
    replaced = [TOKEN_REPLACE.get(t, t) for t in tokens]
    out.add(_norm_space(" ".join(replaced)))

    # 3) kortformer utan "City/Town/United" om det bara är två ord
    if len(tokens) == 2 and tokens[-1] in ("city", "town", "united", "athletic", "rovers", "county"):
        out.add(tokens[0].title())

    return out

# =========================
#  Huvud-funktionen
# =========================
def normalize_team_name(raw_name: str) -> str:
    """
    Robust normaliserare:
      1) Trim + enkla tecken
      2) Manuella alias (TEAM_ALIASES)
      3) Exakta matchningar mot kända kanoniska namn (case-insensitive)
      4) Billiga normalformer → exakt match
      5) Fuzzy-match mot kända kanoniska namn (difflib), tröskel 0.88
    Returnerar originalet trimmat om inget hittas.
    """
    if not isinstance(raw_name, str) or not raw_name.strip():
        return raw_name

    s = _norm_space(raw_name)

    # 1) manuella alias (snabb väg)
    if s in TEAM_ALIASES:
        return TEAM_ALIASES[s]
    if s.title() in TEAM_ALIASES:
        return TEAM_ALIASES[s.title()]

    # 2) kända kanoniska namn (exakt, case-insensitive)
    if _CANONICAL_TEAMS:
        low = _lower_key(s)
        for c in _CANONICAL_TEAMS:
            if _lower_key(c) == low:
                return c

    # 3) prova billiga normalformer
    if _CANONICAL_TEAMS:
        forms = _cheap_normal_forms(s)
        for f in forms:
            lowf = _lower_key(f)
            for c in _CANONICAL_TEAMS:
                if _lower_key(c) == lowf:
                    return c

    # 4) fuzzy (difflib) mot kända kanoniska
    if _CANONICAL_TEAMS:
        target = s.lower()
        best_c, best_r = None, 0.0
        for c in _CANONICAL_TEAMS:
            r = _ratio(target, c.lower())
            if r > best_r:
                best_c, best_r = c, r
        if best_c and best_r >= 0.88:
            return best_c

    # 5) sista fallback: om aliasen delar exakt prefix
    for k, v in TEAM_ALIASES.items():
        if _lower_key(k) == _lower_key(s):
            return v

    return s
