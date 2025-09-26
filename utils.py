import re

TEAM_ALIASES = {
    # Existerande alias
    "Bradford": "Bradford City", "Bradford C": "Bradford City",
    "Cardiff": "Cardiff City", "Cardiff C": "Cardiff City",
    "Man United": "Manchester United", "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "MK Dons": "Milton Keynes Dons",
    "Wolves": "Wolverhampton Wanderers", "Wolverhampton": "Wolverhampton Wanderers",
    "Nott'm Forest": "Nottingham Forest", "Nottm Forest": "Nottingham Forest",
    "Nottingham": "Nottingham Forest",
    "Birmingham": "Birmingham City",
    "Blackburn": "Blackburn Rovers",
    "Bristol C": "Bristol City",
    "Huddersfield": "Huddersfield Town",
    "Leicester": "Leicester City",
    "Norwich": "Norwich City",
    "Stoke": "Stoke City",
    "Swansea": "Swansea City",
    
    # KORRIGERING: Tvinga standardnamnet för problem-lagen
    # Detta garanterar att oavsett om rådatan innehåller "Sheff Wed" eller "Sheffield Wednesday",
    # blir resultatet i datafilen ALLTID "Sheffield Wednesday".
    "Sheff Wed": "Sheffield Wednesday",
    "Sheffield Wed": "Sheffield Wednesday",
    
    "QPR": "Queens Park Rangers",
    "Queens Park Rangers": "Queens Park Rangers",
    
    "Sheff Utd": "Sheffield United",
    "Sheffield Utd": "Sheffield United"
}

def normalize_team_name(raw_name: str) -> str:
    """Standardiserar ett lagnamn med hjälp av en alias-lista."""
    # Säkerställ att input är en sträng
    name = str(raw_name).strip()
    
    # Returnera alias om det finns, annars returnera det ursprungliga namnet.
    # .get() är perfekt för detta.
    return TEAM_ALIASES.get(name, name)
