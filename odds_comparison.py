"""
Odds-jämförelse & Value Betting

Hämtar odds från bookmakers och identifierar value bets där modellens
sannolikheter är högre än oddsen implicerar.
"""
import requests
from bs4 import BeautifulSoup
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def odds_to_probability(odds: float) -> float:
    """
    Konverterar odds till implied probability
    
    Args:
        odds: Decimal odds (t.ex. 2.00)
        
    Returns:
        Implied probability (0-1)
        
    Example:
        2.00 → 0.50 (50%)
        3.00 → 0.33 (33%)
    """
    if odds <= 1.0:
        return 0.0
    return 1.0 / odds


def probability_to_odds(prob: float) -> float:
    """
    Konverterar probability till decimal odds
    
    Args:
        prob: Probability (0-1)
        
    Returns:
        Decimal odds
        
    Example:
        0.50 → 2.00
        0.33 → 3.00
    """
    if prob <= 0.0 or prob >= 1.0:
        return 1.0
    return 1.0 / prob


def calculate_value(model_prob: float, odds: float) -> float:
    """
    Beräknar value för ett spel
    
    Value = (Model Probability * Odds) - 1
    
    Positivt value = Value bet
    Negativt value = Dåligt bet
    
    Args:
        model_prob: Modellens sannolikhet (0-1)
        odds: Bookmaker odds
        
    Returns:
        Value (positivt = bra, negativt = dåligt)
        
    Example:
        Model: 50%, Odds: 2.20 → Value: +0.10 (10% value)
        Model: 50%, Odds: 1.90 → Value: -0.05 (5% negativ value)
    """
    return (model_prob * odds) - 1.0


def is_value_bet(model_prob: float, odds: float, min_value: float = 0.05) -> bool:
    """
    Avgör om ett spel är en value bet
    
    Args:
        model_prob: Modellens sannolikhet
        odds: Bookmaker odds
        min_value: Minsta value för att räknas som value bet
        
    Returns:
        True om value bet
    """
    value = calculate_value(model_prob, odds)
    return value >= min_value


def get_mock_odds(home_team: str, away_team: str) -> Optional[Dict]:
    """
    Hämtar mock odds för testning
    
    I produktion skulle denna funktion scrapa från Oddsportal eller liknande
    
    Args:
        home_team: Hemmalag
        away_team: Bortalag
        
    Returns:
        Dict med odds eller None
    """
    # Mock data för testning
    mock_odds_db = {
        'Arsenal_Liverpool': {'1': 2.10, 'X': 3.40, '2': 3.60},
        'Man City_Chelsea': {'1': 1.50, 'X': 4.50, '2': 6.50},
        'Brighton_Newcastle': {'1': 2.60, 'X': 3.30, '2': 2.80},
        'Tottenham_Man United': {'1': 2.30, 'X': 3.50, '2': 3.10},
        'Fulham_Brentford': {'1': 2.40, 'X': 3.20, '2': 3.00},
    }
    
    key = f"{home_team}_{away_team}"
    
    if key in mock_odds_db:
        return mock_odds_db[key]
    
    # Generera realistiska odds baserat på lagnamn
    # (I verkligheten skulle vi scrapa från bookmakers)
    return {
        '1': 2.20,
        'X': 3.30,
        '2': 3.40
    }


def analyze_value_bets(
    home_team: str,
    away_team: str,
    model_probs: np.ndarray,
    min_value: float = 0.05
) -> Dict:
    """
    Analyserar value bets för en match
    
    Args:
        home_team: Hemmalag
        away_team: Bortalag
        model_probs: Modellens sannolikheter [H, D, A]
        min_value: Minsta value för att räknas som value bet
        
    Returns:
        Dict med analys
    """
    # Hämta odds
    odds = get_mock_odds(home_team, away_team)
    
    if not odds:
        return {
            'has_odds': False,
            'error': 'Kunde inte hämta odds'
        }
    
    # Beräkna implied probabilities från odds
    implied_probs = {
        '1': odds_to_probability(odds['1']),
        'X': odds_to_probability(odds['X']),
        '2': odds_to_probability(odds['2'])
    }
    
    # Beräkna bookmaker margin
    total_implied = sum(implied_probs.values())
    margin = (total_implied - 1.0) * 100  # I procent
    
    # Beräkna value för varje utfall
    values = {
        '1': calculate_value(model_probs[0], odds['1']),
        'X': calculate_value(model_probs[1], odds['X']),
        '2': calculate_value(model_probs[2], odds['2'])
    }
    
    # Identifiera value bets
    value_bets = []
    for outcome in ['1', 'X', '2']:
        if values[outcome] >= min_value:
            value_bets.append({
                'outcome': outcome,
                'model_prob': model_probs[['1', 'X', '2'].index(outcome)],
                'implied_prob': implied_probs[outcome],
                'odds': odds[outcome],
                'value': values[outcome],
                'value_percent': values[outcome] * 100
            })
    
    # Sortera efter value
    value_bets.sort(key=lambda x: x['value'], reverse=True)
    
    return {
        'has_odds': True,
        'home_team': home_team,
        'away_team': away_team,
        'odds': odds,
        'implied_probs': implied_probs,
        'model_probs': {
            '1': model_probs[0],
            'X': model_probs[1],
            '2': model_probs[2]
        },
        'values': values,
        'margin': margin,
        'value_bets': value_bets,
        'has_value': len(value_bets) > 0,
        'best_value': value_bets[0] if value_bets else None
    }


def format_value_analysis(analysis: Dict) -> str:
    """
    Formaterar value-analys till läsbar text
    
    Args:
        analysis: Analys från analyze_value_bets
        
    Returns:
        Formaterad sträng
    """
    if not analysis['has_odds']:
        return "Inga odds tillgängliga"
    
    lines = []
    lines.append(f"Match: {analysis['home_team']} vs {analysis['away_team']}")
    lines.append(f"Bookmaker marginal: {analysis['margin']:.1f}%")
    lines.append("")
    
    lines.append("Odds vs Modell:")
    for outcome in ['1', 'X', '2']:
        odds = analysis['odds'][outcome]
        implied = analysis['implied_probs'][outcome]
        model = analysis['model_probs'][outcome]
        value = analysis['values'][outcome]
        
        outcome_name = {'1': 'Hemma', 'X': 'Oavgjort', '2': 'Borta'}[outcome]
        
        lines.append(f"  {outcome_name} ({outcome}):")
        lines.append(f"    Odds: {odds:.2f} (Implied: {implied:.1%})")
        lines.append(f"    Modell: {model:.1%}")
        lines.append(f"    Value: {value:+.1%}")
    
    lines.append("")
    
    if analysis['has_value']:
        lines.append("✅ VALUE BETS HITTADE:")
        for vb in analysis['value_bets']:
            outcome_name = {'1': 'Hemma', 'X': 'Oavgjort', '2': 'Borta'}[vb['outcome']]
            lines.append(f"  {outcome_name} ({vb['outcome']}): {vb['value_percent']:+.1f}% value @ {vb['odds']:.2f}")
    else:
        lines.append("❌ Inga value bets (alla odds för låga)")
    
    return "\n".join(lines)


def calculate_expected_roi(
    model_prob: float,
    odds: float,
    num_bets: int = 100
) -> float:
    """
    Beräknar förväntad ROI för ett spel
    
    Args:
        model_prob: Modellens sannolikhet
        odds: Odds
        num_bets: Antal spel (för simulering)
        
    Returns:
        Förväntad ROI i procent
    """
    expected_wins = num_bets * model_prob
    expected_losses = num_bets * (1 - model_prob)
    
    total_staked = num_bets
    total_return = expected_wins * odds
    
    roi = ((total_return - total_staked) / total_staked) * 100
    return roi


# Exempel på användning
if __name__ == "__main__":
    print("=== TEST AV ODDS-JÄMFÖRELSE & VALUE BETTING ===\n")
    
    test_matches = [
        ('Arsenal', 'Liverpool', np.array([0.50, 0.30, 0.20])),
        ('Man City', 'Chelsea', np.array([0.70, 0.20, 0.10])),
        ('Brighton', 'Newcastle', np.array([0.35, 0.33, 0.32])),
    ]
    
    for home, away, probs in test_matches:
        print("=" * 80)
        analysis = analyze_value_bets(home, away, probs, min_value=0.05)
        print(format_value_analysis(analysis))
        
        if analysis['has_value']:
            best = analysis['best_value']
            roi = calculate_expected_roi(best['model_prob'], best['odds'])
            print(f"\nFörväntad ROI (100 spel): {roi:+.1f}%")
        
        print()
    
    print("\n=== EXEMPEL PÅ VALUE BETTING ===\n")
    print("Scenario: Modellen ger Arsenal 50% chans, odds är 2.20")
    print()
    
    model_prob = 0.50
    odds = 2.20
    value = calculate_value(model_prob, odds)
    roi = calculate_expected_roi(model_prob, odds, 100)
    
    print(f"Modellens sannolikhet: {model_prob:.0%}")
    print(f"Odds: {odds:.2f}")
    print(f"Implied probability: {odds_to_probability(odds):.1%}")
    print(f"Value: {value:+.1%}")
    print(f"Förväntad ROI (100 spel): {roi:+.1f}%")
    print()
    
    if value > 0:
        print("✅ Detta är en VALUE BET! Spela på Arsenal!")
        print(f"   Om du spelar 100 gånger med 100 kr per gång:")
        print(f"   Förväntad vinst: {roi * 100:.0f} kr")
    else:
        print("❌ Detta är INTE en value bet. Skippa!")
