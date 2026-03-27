# ui_utils.py
import re
from typing import List, Tuple, Optional
import numpy as np

# Importeras från utils.py, som vi vet fungerar
from utils import normalize_team_name
from uncertainty import entropy_norm
from combined_probability import CombinedMatchProbability


def _halfguard_sort_key(probs: Optional[np.ndarray], index: int) -> Tuple[float, float, int]:
    """
    Beräknar sorteringsnyckel för halvgarderingsurval.

    Returnerar (gain, top2, -index) där:
    - gain = second_best probability (marginalnytta av halvgardering)
    - top2 = best + second_best
    - -index = negativt index för deterministisk tie-break (lägst index först)

    Matcher med None/saknade sannolikheter får lägst prioritet (gain=0, top2=0).
    """
    if probs is None:
        return (0.0, 0.0, -index)
    sorted_desc = sorted(probs, reverse=True)
    best = sorted_desc[0]
    second = sorted_desc[1]
    top2 = best + second
    gain = second
    return (gain, top2, -index)

def parse_match_input(text_input: str) -> List[Tuple[str, str]]:
    """
    Parar en text med flera rader av matcher (t.ex. "Arsenal - Chelsea")
    till en lista av normaliserade lagpar.

    Stödjer format:
    - "Arsenal - Chelsea"
    - "Arsenal-Chelsea"
    - "Arsenal vs Chelsea"
    - "Arsenal vs. Chelsea"
    - "Arsenal mot Chelsea"
    - "Arsenal – Chelsea"  (en-dash)
    - "Arsenal — Chelsea"  (em-dash)
    - "1. Arsenal - Chelsea"  (radnummer strippas)
    - "13) Arsenal - Chelsea"
    """
    parsed_matches = []
    errors: List[str] = []
    lines = text_input.strip().split('\n')
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        # Strippa radnummer (t.ex. "1. ", "13) ", "1: ")
        line = re.sub(r'^\d+[.):\s]+\s*', '', line).strip()
        if not line:
            continue

        # Försök olika separatorer (mer flexibel regex)
        # Matchar: " - ", "-", " – ", " — ", " vs ", " vs. ", " mot ", etc.
        parts = re.split(r'\s*[-–—]\s*|\s+vs\.?\s+|\s+mot\s+', line, maxsplit=1, flags=re.IGNORECASE)

        if len(parts) == 2:
            home_raw, away_raw = parts[0].strip(), parts[1].strip()
            if home_raw and away_raw:
                home_team = normalize_team_name(home_raw)
                away_team = normalize_team_name(away_raw)
                if home_team and away_team:
                    parsed_matches.append((home_team, away_team))
                else:
                    errors.append(f"Rad {line_num}: kunde inte normalisera lagnamn i '{line}'")
            else:
                errors.append(f"Rad {line_num}: tomt lagnamn i '{line}'")
        else:
            errors.append(f"Rad {line_num}: kunde inte tolka '{line}' — förväntat format: 'Hemmalag - Bortalag'")

    return parsed_matches


def parse_match_input_with_errors(text_input: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Samma som parse_match_input men returnerar även felmeddelanden.

    Returnerar (matches, errors) där errors är en lista av strängar
    som beskriver rader som inte kunde tolkas.
    """
    parsed_matches = []
    errors: List[str] = []
    lines = text_input.strip().split('\n')
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        # Strippa radnummer (t.ex. "1. ", "13) ", "1: ")
        line = re.sub(r'^\d+[.):\s]+\s*', '', line).strip()
        if not line:
            continue

        parts = re.split(r'\s*[-–—]\s*|\s+vs\.?\s+|\s+mot\s+', line, maxsplit=1, flags=re.IGNORECASE)

        if len(parts) == 2:
            home_raw, away_raw = parts[0].strip(), parts[1].strip()
            if home_raw and away_raw:
                home_team = normalize_team_name(home_raw)
                away_team = normalize_team_name(away_raw)
                if home_team and away_team:
                    parsed_matches.append((home_team, away_team))
                else:
                    errors.append(f"Rad {line_num}: kunde inte normalisera lagnamn i '{line}'")
            else:
                errors.append(f"Rad {line_num}: tomt lagnamn i '{line}'")
        else:
            errors.append(f"Rad {line_num}: kunde inte tolka '{line}' — förväntat format: 'Hemmalag - Bortalag'")

    return parsed_matches, errors

# --- NYTT: Logik för Halvgarderingar ---

def pick_half_guards(match_probs: List[Optional[np.ndarray]], n_guards: int) -> List[int]:
    """
    Väljer ut matcher för halvgardering baserat på marginalnytta (gain).

    Strategi: Välj matcher där en halvgardering ger störst ökning i
    träffsannolikhet, dvs. högst gain = second_best probability.

    Sortering:
    1. Högst gain (second_best) först
    2. Vid lika: högst top2 (best + second_best)
    3. Vid fortsatt lika: lägst index (deterministisk tie-break)

    Matcher som saknar data (None) får lägst prioritet.

    Returnerar en lista med index för de matcher som ska halvgarderas.
    """
    if n_guards <= 0:
        return []

    indexed = [(i, probs) for i, probs in enumerate(match_probs)]
    indexed.sort(key=lambda x: _halfguard_sort_key(x[1], x[0]), reverse=True)

    return [i for i, _ in indexed[:n_guards]]


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


# --- Kombinerade halvgarderingar (odds + modell + streck) ---

def pick_half_guards_combined(
    combined_matches: List[CombinedMatchProbability],
    n_guards: int,
) -> List[int]:
    """
    Väljer halvgarderingar baserat på marginalnytta (gain) från
    den kombinerade sannolikheten (odds + modell + streck).

    Strategi:
    1. Beräkna gain = second_best probability från cm.probs
    2. Sortera på högst gain först
    3. Vid lika: högst top2 (best + second_best)
    4. Vid fortsatt lika: lägst index (deterministisk tie-break)

    Returnerar lista med index.
    """
    if n_guards <= 0:
        return []

    indexed = [(i, cm.probs) for i, cm in enumerate(combined_matches)]
    indexed.sort(key=lambda x: _halfguard_sort_key(x[1], x[0]), reverse=True)
    return [i for i, _ in indexed[:n_guards]]


def get_halfguard_sign_combined(cm: CombinedMatchProbability) -> str:
    """
    Väljer halvgarderingstecken baserat på kombinerad sannolikhet.

    Tar bort det minst sannolika utfallet enligt den kombinerade
    sannolikheten (inte bara modellen).
    """
    signs = ["1", "X", "2"]
    probs = cm.probs
    least_likely = int(np.argmin(probs))
    return "".join(signs[i] for i in range(3) if i != least_likely)
