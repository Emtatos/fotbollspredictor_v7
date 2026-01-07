"""
Confidence Score & Säkerhetsfilter

Beräknar hur säker modellen är på en prediktion och filtrerar osäkra matcher.
"""
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_confidence(probs: np.ndarray) -> float:
    """
    Beräknar confidence score för en prediktion
    
    Confidence = (Max prob - Second max prob) / Max prob
    
    Högt värde = Tydlig favorit
    Lågt värde = Jämnt, osäkert
    
    Args:
        probs: Sannolikheter [H, D, A]
        
    Returns:
        Confidence score mellan 0 och 1
        
    Examples:
        [0.60, 0.25, 0.15] → 0.58 (Hög confidence)
        [0.35, 0.33, 0.32] → 0.06 (Låg confidence)
    """
    if len(probs) < 2:
        return 0.0
    
    sorted_probs = np.sort(probs)[::-1]  # Sortera fallande
    max_prob = sorted_probs[0]
    second_prob = sorted_probs[1]
    
    if max_prob == 0:
        return 0.0
    
    confidence = (max_prob - second_prob) / max_prob
    return confidence


def get_confidence_level(confidence: float) -> str:
    """
    Konverterar confidence score till läsbar nivå
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Nivå som sträng
    """
    if confidence >= 0.5:
        return "Mycket hög"
    elif confidence >= 0.3:
        return "Hög"
    elif confidence >= 0.15:
        return "Medel"
    elif confidence >= 0.05:
        return "Låg"
    else:
        return "Mycket låg"


def should_bet(probs: np.ndarray, min_confidence: float = 0.15) -> Tuple[bool, str]:
    """
    Avgör om man bör spela på en match baserat på confidence
    
    Args:
        probs: Sannolikheter [H, D, A]
        min_confidence: Minsta confidence för att spela
        
    Returns:
        Tuple med (ska spela, anledning)
    """
    confidence = calculate_confidence(probs)
    
    if confidence >= min_confidence:
        return True, f"Confidence {confidence:.2f} är tillräckligt hög"
    else:
        return False, f"Confidence {confidence:.2f} är för låg (minimum {min_confidence:.2f})"


def get_recommended_bet_type(probs: np.ndarray, confidence: float) -> str:
    """
    Rekommenderar typ av spel baserat på confidence
    
    Args:
        probs: Sannolikheter [H, D, A]
        confidence: Confidence score
        
    Returns:
        Rekommenderad speltyp
    """
    max_idx = np.argmax(probs)
    max_prob = probs[max_idx]
    
    if confidence >= 0.5:
        # Mycket hög confidence → Singel
        outcomes = ['1', 'X', '2']
        return f"Singel ({outcomes[max_idx]})"
    
    elif confidence >= 0.3:
        # Hög confidence → Singel eller dubbelchans
        if max_idx == 0:  # Hemmavinst
            return "Singel (1) eller Dubbelchans (1X)"
        elif max_idx == 2:  # Bortavinst
            return "Singel (2) eller Dubbelchans (X2)"
        else:  # Oavgjort
            return "Dubbelchans (1X eller X2)"
    
    elif confidence >= 0.15:
        # Medel confidence → Dubbelchans
        if max_idx == 0:
            return "Dubbelchans (1X)"
        elif max_idx == 2:
            return "Dubbelchans (X2)"
        else:
            return "Dubbelchans (1X eller X2)"
    
    else:
        # Låg confidence → Skippa
        return "Skippa denna match (för osäker)"


def filter_matches_by_confidence(
    matches: list,
    min_confidence: float = 0.15
) -> Tuple[list, list]:
    """
    Filtrerar matcher baserat på confidence
    
    Args:
        matches: Lista med matcher och deras sannolikheter
        min_confidence: Minsta confidence för att inkludera
        
    Returns:
        Tuple med (godkända matcher, avvisade matcher)
    """
    approved = []
    rejected = []
    
    for match in matches:
        probs = match.get('probs', np.array([0.33, 0.33, 0.34]))
        confidence = calculate_confidence(probs)
        
        match_with_confidence = {**match, 'confidence': confidence}
        
        if confidence >= min_confidence:
            approved.append(match_with_confidence)
        else:
            rejected.append(match_with_confidence)
    
    # Sortera efter confidence (högst först)
    approved.sort(key=lambda x: x['confidence'], reverse=True)
    rejected.sort(key=lambda x: x['confidence'], reverse=True)
    
    return approved, rejected


def get_confidence_stats(matches: list) -> dict:
    """
    Beräknar statistik över confidence för en lista med matcher
    
    Args:
        matches: Lista med matcher
        
    Returns:
        Dict med statistik
    """
    if not matches:
        return {
            'count': 0,
            'mean': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0
        }
    
    confidences = [m.get('confidence', 0.0) for m in matches]
    
    return {
        'count': len(confidences),
        'mean': np.mean(confidences),
        'median': np.median(confidences),
        'min': np.min(confidences),
        'max': np.max(confidences),
        'high_confidence': sum(1 for c in confidences if c >= 0.3),
        'medium_confidence': sum(1 for c in confidences if 0.15 <= c < 0.3),
        'low_confidence': sum(1 for c in confidences if c < 0.15)
    }


# Exempel på användning
if __name__ == "__main__":
    print("=== TEST AV CONFIDENCE SCORE ===\n")
    
    test_cases = [
        ([0.60, 0.25, 0.15], "Tydlig favorit"),
        ([0.50, 0.30, 0.20], "Klar favorit"),
        ([0.45, 0.35, 0.20], "Svag favorit"),
        ([0.40, 0.35, 0.25], "Osäker match"),
        ([0.35, 0.33, 0.32], "Mycket osäker"),
    ]
    
    for probs, description in test_cases:
        probs_array = np.array(probs)
        confidence = calculate_confidence(probs_array)
        level = get_confidence_level(confidence)
        should, reason = should_bet(probs_array)
        bet_type = get_recommended_bet_type(probs_array, confidence)
        
        print(f"{description}:")
        print(f"  Sannolikheter: {probs}")
        print(f"  Confidence: {confidence:.3f} ({level})")
        print(f"  Spela: {'✅ Ja' if should else '❌ Nej'}")
        print(f"  Rekommendation: {bet_type}")
        print()
    
    # Testa filtrering
    print("\n=== TEST AV FILTRERING ===\n")
    
    test_matches = [
        {'home': 'Arsenal', 'away': 'Liverpool', 'probs': np.array([0.60, 0.25, 0.15])},
        {'home': 'Man City', 'away': 'Chelsea', 'probs': np.array([0.50, 0.30, 0.20])},
        {'home': 'Brighton', 'away': 'Newcastle', 'probs': np.array([0.35, 0.33, 0.32])},
    ]
    
    approved, rejected = filter_matches_by_confidence(test_matches, min_confidence=0.15)
    
    print(f"Godkända matcher: {len(approved)}")
    for match in approved:
        print(f"  {match['home']} vs {match['away']}: Confidence {match['confidence']:.3f}")
    
    print(f"\nAvvisade matcher: {len(rejected)}")
    for match in rejected:
        print(f"  {match['home']} vs {match['away']}: Confidence {match['confidence']:.3f}")
    
    # Statistik
    stats = get_confidence_stats(test_matches)
    print(f"\nStatistik:")
    print(f"  Medel confidence: {stats['mean']:.3f}")
    print(f"  Hög confidence: {stats['high_confidence']}")
    print(f"  Medel confidence: {stats['medium_confidence']}")
    print(f"  Låg confidence: {stats['low_confidence']}")
