# ui_utils.py
import math
import re
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# Importeras från utils.py, som vi vet fungerar
from utils import normalize_team_name
from uncertainty import entropy_norm

# Epsilon för log-beräkningar (undviker log(0))
EPS = 1e-15

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

def compute_half_guard_gain(probs: np.ndarray) -> float:
    """
    Beräknar gain från att halvgardera en match.
    
    gain = log(P_half) - log(P_spik)
    
    där P_spik = p_top1 (högsta sannolikheten)
    och P_half = p_top1 + p_top2 (summan av de två högsta)
    
    Detta mäter hur mycket kupongens log-sannolikhet förbättras
    av att halvgardera just den matchen.
    
    Args:
        probs: Sannolikheter [p1, pX, p2]
    
    Returns:
        gain (float): Förbättring i log-sannolikhet
    """
    sorted_probs = np.sort(probs)[::-1]
    p_spik = max(sorted_probs[0], EPS)
    p_half = max(sorted_probs[0] + sorted_probs[1], EPS)
    
    gain = math.log(p_half) - math.log(p_spik)
    return gain


def pick_half_guards(
    match_probs: List[Optional[np.ndarray]], 
    n_guards: int,
    entropy_values: Optional[List[Optional[float]]] = None,
    trust_scores: Optional[List[Optional[int]]] = None,
) -> List[int]:
    """
    Väljer ut matcher för halvgardering baserat på expected gain.
    
    Strategi: Välj de N_HALF matcher som ger störst förbättring av
    kupongens log-sannolikhet vid övergång från spik till halv.
    
    Matcher som saknar data (None) får högsta prioritet.
    
    Tie-breakers om gain är lika/nära:
    1. högre entropy_norm först
    2. lägre trust_score först (sämre datatäckning prioriteras)
    3. lägre p_top1 först (svagare spik prioriteras)

    Args:
        match_probs: Lista med sannolikheter [1, X, 2] per match (eller None)
        n_guards: Antal halvgarderingar att välja
        entropy_values: Valfri lista med entropy per match (för tie-breaker)
        trust_scores: Valfri lista med trust score per match (för tie-breaker)

    Returns:
        Lista med index för de matcher som ska halvgarderas.
    """
    if n_guards <= 0:
        return []

    scored_matches = []
    for i, probs in enumerate(match_probs):
        if probs is None:
            scored_matches.append({
                'index': i,
                'gain': float('inf'),
                'entropy': 2.0,
                'trust_score': 0,
                'p_spik': 0.0,
            })
        else:
            gain = compute_half_guard_gain(probs)
            entropy = entropy_values[i] if entropy_values and entropy_values[i] is not None else entropy_norm(probs[0], probs[1], probs[2])
            trust = trust_scores[i] if trust_scores and trust_scores[i] is not None else 50
            p_spik = float(np.max(probs))
            
            scored_matches.append({
                'index': i,
                'gain': gain,
                'entropy': entropy,
                'trust_score': trust,
                'p_spik': p_spik,
            })

    # Sortera: högst gain först, sedan högst entropy, sedan lägst trust, sedan lägst p_spik
    scored_matches.sort(
        key=lambda x: (
            -x['gain'],
            -x['entropy'],
            x['trust_score'],
            x['p_spik'],
        )
    )
    
    # Välj ut de n_guards bästa indexen
    guard_indices = [m['index'] for m in scored_matches[:n_guards]]
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
