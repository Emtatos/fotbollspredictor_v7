import re

# Steg 1: Definiera kanoniska namn (det fullständiga, korrekta namnet) och alla dess kända alias.
TEAM_ALIASES = {
    "Sheffield Wednesday": {
        "Sheffield Weds", # <<< DEN FELANDE LÄNKEN!
        "Sheffield Wed", 
        "Sheff Wed"
    },
    "Queens Park Rangers": {
        "QPR", 
        "Queens Park Rangers", 
        "Queen's Park Rangers"
    },
    "Manchester United": {"Man United", "Man Utd", "Manchester United"},
    "Manchester City": {"Man City", "Manchester City"},
    "Nottingham Forest": {"Nott'm Forest", "Nottm Forest", "Nottingham", "Nottingham Forest"},
    "Wolverhampton Wanderers": {"Wolves", "Wolverhampton", "Wolverhampton Wanderers"},
    "Sheffield United": {"Sheff Utd", "Sheffield Utd", "Sheffield United"},
    "West Bromwich Albion": {"West Brom", "West Bromwich", "West Bromwich Albion"},
    "Birmingham City": {"Birmingham", "Birmingham City"},
    "Blackburn Rovers": {"Blackburn", "Blackburn Rovers"},
    "Bristol City": {"Bristol C", "Bristol City"},
    "Cardiff City": {"Cardiff", "Cardiff C", "Cardiff City"},
    "Huddersfield Town": {"Huddersfield", "Huddersfield Town"},
    "Leicester City": {"Leicester", "Leicester City"},
    "Norwich City": {"Norwich", "Norwich City"},
    "Stoke City": {"Stoke", "Stoke City"},
    "Swansea City": {"Swansea", "Swansea City"},
    "Bradford City": {"Bradford", "Bradford C"},
    "Millwall": {"Millwall"},"West Bromwich Albion": {
    "West Brom",
    "West Bromwich",
    "West Bromwich Albion"
},
    "MK Dons": {"MK Dons"} # Kanoniska namnet kan vara ett alias till sig själv
}

# Steg 2: Skapa en "omvänd" dictionary för snabb uppslagning.
ALIAS_TO_CANONICAL = {
    alias.lower(): canonical_name
    for canonical_name, aliases in TEAM_ALIASES.items()
    # Lägg till det kanoniska namnet som ett alias till sig själv för fullständig robusthet
    for alias in list(aliases) + [canonical_name]
}

def normalize_team_name(raw_name: str) -> str:
    """
    Standardiserar ett lagnamn till sin kanoniska form.
    Funktionen är inte skiftlägeskänslig (case-insensitive).
    """
    if not isinstance(raw_name, str):
        return raw_name

    normalized_key = " ".join(raw_name.strip().split()).lower()
    
    return ALIAS_TO_CANONICAL.get(normalized_key, raw_name.strip())
