# ui_utils.py
import re
from typing import List, Tuple, Optional
import numpy as np

# Importeras från utils.py, som vi vet fungerar
from utils import normalize_team_name

def parse_match_input(text_input: str) -> List[Tuple[str, str]]:
    """
    Parar en text med flera rader av matcher (t.ex. "Arsenal - Chelsea")
    till en lista av normaliserade lagpar.
    
    Stödjer format:
    - "Arsenal - Chelsea"
    - "Arsenal-Chelsea" 
    - "Arsenal vs Chelsea"
    - "Arsenal mot Chelsea"
    """
    parsed_matches = []
    lines = text_input.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Försök olika separatorer (mer flexibel regex)
        # Matchar: " - ", "-", " vs ", " mot ", etc.
        parts = re.split(r'\s*[-–—]\s*|\s+vs\.?\s+|\s+mot\s+', line, maxsplit=1, flags=re.IGNORECASE)
        
        if len(parts) == 2:
            home_raw, away_raw = parts[0].strip(), parts[1].strip()
            if home_raw and away_raw:
                home_team = normalize_team_name(home_raw)
                away_team = normalize_team_name(away_raw)
                if home_team and away_team:
                    parsed_matches.append((home_team, away_team))
    
    return parsed_matches

# --- NYTT: Logik för Halvgarderingar ---

def pick_half_guards(match_probs: List[Optional[np.ndarray]], n_guards: int) -> List[int]:
    """
    Väljer ut de mest osäkra matcherna för halvgardering.

    Strategi: Välj matcher där skillnaden mellan bästa och näst bästa utfall är minst.
    Matcher som saknar data (None) får högsta prioritet för gardering.

    Returnerar en lista med index för de matcher som ska halvgarderas.
    """
    if n_guards <= 0:
        return []

    scored_matches = []
    for i, probs in enumerate(match_probs):
        if probs is None:
            # Ge matcher utan data högsta prioritet (simulera en marginal på -1)
            margin = -1.0 
        else:
            # Sortera sannolikheterna och beräkna marginalen
            sorted_probs = sorted(probs, reverse=True)
            margin = sorted_probs[0] - sorted_probs[1]
        
        scored_matches.append({'margin': margin, 'index': i})

    # Sortera efter marginal (lägst först, dvs. osäkrast först)
    scored_matches.sort(key=lambda x: x['margin'])
    
    # Välj ut de n_guards bästa indexen
    guard_indices = [match['index'] for match in scored_matches[:n_guards]]
    return guard_indices


def get_halfguard_sign(probs: np.ndarray) -> str:
    """
    Returnerar halvgarderingstecknet (t.ex. "1X") genom att
    ta bort det minst sannolika utfallet.
    """
    signs = ['1', 'X', '2']
    # Hitta index för det minst sannolika utfallet
    least_likely_index = np.argmin(probs)
    
    # Bygg en sträng med de två återstående tecknen
    return "".join([signs[i] for i in range(3) if i != least_likely_index])
