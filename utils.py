# utils.py
import re

TEAM_ALIASES = {
    "Bradford": "Bradford City", "Bradford C": "Bradford City",
    "Cardiff": "Cardiff City", "Cardiff C": "Cardiff City",
    "Man United": "Manchester United", "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Sheffield Wed": "Sheffield Wednesday", "Sheff Wed": "Sheffield Wednesday",
    "Sheffield Utd": "Sheffield United", "Sheff Utd": "Sheffield United",
    "QPR": "Queens Park Rangers",
    "MK Dons": "Milton Keynes Dons",
    "Wolves": "Wolverhampton Wanderers", "Wolverhampton": "Wolverhampton Wanderers",
    "Nott'm Forest": "Nottingham Forest", "Nottm Forest": "Nottingham Forest",
}

def normalize_team_name(raw_name: str) -> str:
    """Standardiserar ett lagnamn med hjälp av en alias-lista."""
    name = str(raw_name).strip()
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    return TEAM_ALIASES.get(name, name)
