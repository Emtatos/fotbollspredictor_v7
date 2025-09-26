import re

TEAM_ALIASES = {
    # Existerande alias
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

    # NYA TILLÄGG för att hantera kortformer
    "Nottingham": "Nottingham Forest",
    "Birmingham": "Birmingham City",
    "Blackburn": "Blackburn Rovers",
    "Bristol C": "Bristol City",
    "Huddersfield": "Huddersfield Town",
    "Leicester": "Leicester City",
    "Norwich": "Norwich City",
    "Stoke": "Stoke City",
    "Swansea": "Swansea City"
}

def normalize_team_name(raw_name: str) -> str:
    """Standardiserar ett lagnamn med hjälp av en alias-lista."""
    # Säkerställ att input är en sträng
    name = str(raw_name).strip()
    
    # Om en direkt matchning finns, använd den
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    
    # Annars, returnera den ursprungliga strängen (eller en matchning om den finns, vilket är redundant men ofarligt)
    return TEAM_ALIASES.get(name, name)
