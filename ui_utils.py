# ui_utils.py
import re
from typing import List, Tuple, Optional
import numpy as np

# Importeras från utils.py, som vi vet fungerar
from utils import normalize_team_name
from uncertainty import entropy_norm

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
    Väljer ut de mest osäkra matcherna för halvgardering baserat på entropy.

    Strategi: Välj matcher med högst entropy (osäkerhet över hela 1/X/2-fördelningen).
    Matcher som saknar data (None) får högsta prioritet för gardering (entropy = 2.0).

    Returnerar en lista med index för de matcher som ska halvgarderas.
    """
    if n_guards <= 0:
        return []

    scored_matches = []
    for i, probs in enumerate(match_probs):
        if probs is None:
            # Ge matcher utan data högsta prioritet (entropy > 1.0)
            entropy = 2.0
        else:
            # Beräkna normaliserad entropy för sannolikhetsfördelningen
            entropy = entropy_norm(probs[0], probs[1], probs[2])
        
        scored_matches.append({'entropy': entropy, 'index': i})

    # Sortera efter entropy (högst först, dvs. mest osäker först)
    scored_matches.sort(key=lambda x: x['entropy'], reverse=True)
    
    # Välj ut de n_guards bästa indexen
    guard_indices = [match['index'] for match in scored_matches[:n_guards]]
    return guard_indices


def calculate_match_entropy(probs: Optional[np.ndarray]) -> Optional[float]:
    """
    Beräknar entropy för en match.

    Args:
        probs: Sannolikheter [1, X, 2] eller None

    Returns:
        Normaliserad entropy (0-1) eller None om probs är None
    """
    if probs is None:
        return None
    return entropy_norm(probs[0], probs[1], probs[2])


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
