"""
Poisson-distribution för Målprediktion

Förutspår exakt antal mål och ger tips på målmarknader som:
- Över/Under 2.5 mål
- BTTS (Both Teams To Score)
- Exakt resultat (0-0, 1-0, 2-1, etc.)
"""
import numpy as np
from scipy.stats import poisson
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


def calculate_expected_goals(
    team_goals_for: float,
    team_goals_against: float,
    opponent_goals_for: float,
    opponent_goals_against: float,
    home_advantage: float = 0.3
) -> Tuple[float, float]:
    """
    Beräknar förväntade mål för hemma- och bortalag
    
    Args:
        team_goals_for: Hemmalags genomsnittliga gjorda mål
        team_goals_against: Hemmalags genomsnittliga insläppta mål
        opponent_goals_for: Bortalags genomsnittliga gjorda mål
        opponent_goals_against: Bortalags genomsnittliga insläppta mål
        home_advantage: Hemmaplansfördel (extra mål)
        
    Returns:
        Tuple med (hemma förväntade mål, borta förväntade mål)
    """
    # Hemmalags attack vs Bortalags försvar
    home_expected = (team_goals_for + opponent_goals_against) / 2 + home_advantage
    
    # Bortalags attack vs Hemmalags försvar
    away_expected = (opponent_goals_for + team_goals_against) / 2
    
    return home_expected, away_expected


def poisson_probability(expected_goals: float, actual_goals: int) -> float:
    """
    Beräknar sannolikheten för ett specifikt antal mål
    
    Args:
        expected_goals: Förväntade mål (lambda)
        actual_goals: Faktiska mål
        
    Returns:
        Sannolikhet (0-1)
    """
    return poisson.pmf(actual_goals, expected_goals)


def calculate_score_probabilities(
    home_expected: float,
    away_expected: float,
    max_goals: int = 6
) -> Dict[Tuple[int, int], float]:
    """
    Beräknar sannolikheter för alla möjliga resultat
    
    Args:
        home_expected: Hemmalags förväntade mål
        away_expected: Bortalags förväntade mål
        max_goals: Maximalt antal mål att beräkna
        
    Returns:
        Dict med (hemma_mål, borta_mål) → sannolikhet
    """
    probabilities = {}
    
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob_home = poisson_probability(home_expected, home_goals)
            prob_away = poisson_probability(away_expected, away_goals)
            prob_score = prob_home * prob_away
            probabilities[(home_goals, away_goals)] = prob_score
    
    return probabilities


def predict_match_outcome(score_probs: Dict[Tuple[int, int], float]) -> Dict[str, float]:
    """
    Beräknar sannolikheter för 1X2 baserat på resultat
    
    Args:
        score_probs: Sannolikheter för alla resultat
        
    Returns:
        Dict med sannolikheter för H, D, A
    """
    prob_home = sum(prob for (h, a), prob in score_probs.items() if h > a)
    prob_draw = sum(prob for (h, a), prob in score_probs.items() if h == a)
    prob_away = sum(prob for (h, a), prob in score_probs.items() if h < a)
    
    return {
        'home_win': prob_home,
        'draw': prob_draw,
        'away_win': prob_away
    }


def predict_over_under(
    home_expected: float,
    away_expected: float,
    threshold: float = 2.5
) -> Dict[str, float]:
    """
    Beräknar sannolikheter för Över/Under mål
    
    Args:
        home_expected: Hemmalags förväntade mål
        away_expected: Bortalags förväntade mål
        threshold: Tröskelvärde för mål
        
    Returns:
        Dict med sannolikheter för över/under
    """
    total_expected = home_expected + away_expected
    
    # Beräkna sannolikhet för varje totalt antal mål
    prob_under = 0.0
    for goals in range(int(threshold) + 1):
        prob_under += poisson_probability(total_expected, goals)
    
    prob_over = 1.0 - prob_under
    
    return {
        'over': prob_over,
        'under': prob_under,
        'expected_total': total_expected
    }


def predict_btts(
    home_expected: float,
    away_expected: float
) -> Dict[str, float]:
    """
    Beräknar sannolikhet för BTTS (Both Teams To Score)
    
    Args:
        home_expected: Hemmalags förväntade mål
        away_expected: Bortalags förväntade mål
        
    Returns:
        Dict med sannolikheter för BTTS ja/nej
    """
    # Sannolikhet att hemmalaget inte gör mål
    prob_home_zero = poisson_probability(home_expected, 0)
    
    # Sannolikhet att bortalaget inte gör mål
    prob_away_zero = poisson_probability(away_expected, 0)
    
    # Sannolikhet att minst ett lag inte gör mål
    prob_no_btts = prob_home_zero + prob_away_zero - (prob_home_zero * prob_away_zero)
    
    # Sannolikhet att båda lagen gör mål
    prob_btts = 1.0 - prob_no_btts
    
    return {
        'btts_yes': prob_btts,
        'btts_no': prob_no_btts
    }


def get_most_likely_scores(
    score_probs: Dict[Tuple[int, int], float],
    top_n: int = 5
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Hämtar de mest sannolika resultaten
    
    Args:
        score_probs: Sannolikheter för alla resultat
        top_n: Antal resultat att returnera
        
    Returns:
        Lista med (resultat, sannolikhet) sorterad efter sannolikhet
    """
    sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:top_n]


def analyze_match_goals(
    home_team: str,
    away_team: str,
    home_goals_for: float,
    home_goals_against: float,
    away_goals_for: float,
    away_goals_against: float
) -> Dict:
    """
    Komplett målanalys för en match
    
    Args:
        home_team: Hemmalag
        away_team: Bortalag
        home_goals_for: Hemmalags genomsnittliga gjorda mål
        home_goals_against: Hemmalags genomsnittliga insläppta mål
        away_goals_for: Bortalags genomsnittliga gjorda mål
        away_goals_against: Bortalags genomsnittliga insläppta mål
        
    Returns:
        Dict med komplett analys
    """
    # Beräkna förväntade mål
    home_exp, away_exp = calculate_expected_goals(
        home_goals_for, home_goals_against,
        away_goals_for, away_goals_against
    )
    
    # Beräkna resultat-sannolikheter
    score_probs = calculate_score_probabilities(home_exp, away_exp)
    
    # Beräkna 1X2
    outcome_probs = predict_match_outcome(score_probs)
    
    # Beräkna Över/Under
    over_under = predict_over_under(home_exp, away_exp, 2.5)
    
    # Beräkna BTTS
    btts = predict_btts(home_exp, away_exp)
    
    # Mest sannolika resultat
    likely_scores = get_most_likely_scores(score_probs, 5)
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'expected_goals': {
            'home': home_exp,
            'away': away_exp,
            'total': home_exp + away_exp
        },
        'outcome_probs': outcome_probs,
        'over_under_2_5': over_under,
        'btts': btts,
        'most_likely_scores': likely_scores
    }


def format_goal_analysis(analysis: Dict) -> str:
    """
    Formaterar målanalys till läsbar text
    
    Args:
        analysis: Analys från analyze_match_goals
        
    Returns:
        Formaterad sträng
    """
    lines = []
    lines.append(f"Match: {analysis['home_team']} vs {analysis['away_team']}")
    lines.append("")
    
    # Förväntade mål
    exp = analysis['expected_goals']
    lines.append(f"Förväntade mål:")
    lines.append(f"  {analysis['home_team']}: {exp['home']:.2f}")
    lines.append(f"  {analysis['away_team']}: {exp['away']:.2f}")
    lines.append(f"  Totalt: {exp['total']:.2f}")
    lines.append("")
    
    # 1X2
    outcome = analysis['outcome_probs']
    lines.append(f"1X2 (från Poisson):")
    lines.append(f"  Hemmavinst: {outcome['home_win']:.1%}")
    lines.append(f"  Oavgjort: {outcome['draw']:.1%}")
    lines.append(f"  Bortavinst: {outcome['away_win']:.1%}")
    lines.append("")
    
    # Över/Under
    ou = analysis['over_under_2_5']
    lines.append(f"Över/Under 2.5 mål:")
    lines.append(f"  Över 2.5: {ou['over']:.1%} {'✅' if ou['over'] > 0.5 else ''}")
    lines.append(f"  Under 2.5: {ou['under']:.1%} {'✅' if ou['under'] > 0.5 else ''}")
    lines.append("")
    
    # BTTS
    btts = analysis['btts']
    lines.append(f"Both Teams To Score:")
    lines.append(f"  Ja: {btts['btts_yes']:.1%} {'✅' if btts['btts_yes'] > 0.5 else ''}")
    lines.append(f"  Nej: {btts['btts_no']:.1%} {'✅' if btts['btts_no'] > 0.5 else ''}")
    lines.append("")
    
    # Mest sannolika resultat
    lines.append(f"Mest sannolika resultat:")
    for (h, a), prob in analysis['most_likely_scores']:
        lines.append(f"  {h}-{a}: {prob:.1%}")
    
    return "\n".join(lines)


# Exempel på användning
if __name__ == "__main__":
    print("=== TEST AV POISSON MÅLPREDIKTION ===\n")
    
    test_matches = [
        ('Arsenal', 'Liverpool', 1.8, 0.8, 1.6, 0.9),
        ('Man City', 'Chelsea', 2.2, 0.6, 1.4, 1.0),
        ('Brighton', 'Newcastle', 1.2, 1.2, 1.3, 1.1),
    ]
    
    for home, away, hgf, hga, agf, aga in test_matches:
        print("=" * 80)
        analysis = analyze_match_goals(home, away, hgf, hga, agf, aga)
        print(format_goal_analysis(analysis))
        print()
    
    print("\n=== JÄMFÖRELSE MED XGBOOST ===\n")
    print("Poisson ger ofta bättre prediktioner för målmarknader:")
    print("- Över/Under 2.5 mål")
    print("- BTTS (Both Teams To Score)")
    print("- Exakt resultat")
    print()
    print("XGBoost är bättre för 1X2-marknaden")
    print()
    print("Rekommendation: Använd båda modellerna tillsammans!")
