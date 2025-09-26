import re

# Steg 1: Definiera kanoniska namn och alla deras kända alias i en dictionary.
# Det kanoniska namnet (nyckeln) är den exakta strängen vi fann i vår datafil.
TEAM_ALIASES = {
    "Sheffield Wed": {
        "Sheffield Wed", 
        "Sheff Wed", 
        "Sheffield Wednesday"
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
    "Swansea City": {"Swansea", "Swansea City"}
}

# Steg 2: Skapa en "omvänd" dictionary för snabb uppslagning.
# Den mappar varje alias till sitt kanoniska namn.
ALIAS_TO_CANONICAL = {
    alias.lower(): canonical_name
    for canonical_name, aliases in TEAM_ALIASES.items()
    for alias in aliases
}

def normalize_team_name(raw_name: str) -> str:
    """
    Standardiserar ett lagnamn till sin kanoniska form.
    Funktionen är inte skiftlägeskänslig (case-insensitive).
    """
    if not isinstance(raw_name, str):
        return raw_name

    # Normalisera namnet genom att ta bort extra mellanslag och göra det till gemener
    # Detta gör matchningen robust mot skiftlägesfel.
    normalized_key = " ".join(raw_name.strip().split()).lower()
    
    # Hitta det kanoniska namnet i vår uppslags-dictionary.
    # Om det inte finns, returnera det ursprungliga, rensade namnet.
    return ALIAS_TO_CANONICAL.get(normalized_key, raw_name.strip())
