# ui_utils.py
import re
from typing import List, Tuple
from utils import normalize_team_name

def parse_match_input(text_input: str) -> List[Tuple[str, str]]:
    """
    Parsar en text med flera rader av matcher (t.ex. 'Arsenal - Chelsea')
    till en lista av normaliserade lagpar.
    Tillåtna separatorer: ' - ', ' vs ', ' mot ' (med valfria mellanrum).
    """
    parsed_matches = []
    if not text_input:
        return parsed_matches

    lines = text_input.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = re.split(r'\s+-\s+|\s+vs\s+|\s+mot\s+', line, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            home_raw, away_raw = parts
            home_team = normalize_team_name(home_raw)
            away_team = normalize_team_name(away_raw)
            if home_team and away_team and home_team != away_team:
                parsed_matches.append((home_team, away_team))

    return parsed_matches
