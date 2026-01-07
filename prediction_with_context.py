"""
Prediktionsfunktion som integrerar AI-baserad matchkontext
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

try:
    from news_scraper_v2 import get_match_context
    HAS_CONTEXT = True
except ImportError:
    HAS_CONTEXT = False


def adjust_probabilities_with_context(
    probs: np.ndarray,
    home_team: str,
    away_team: str,
    use_context: bool = True
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Justerar sannolikheter baserat på aktuell matchkontext (skador, form, nyheter)
    
    Args:
        probs: Ursprungliga sannolikheter [H, D, A]
        home_team: Hemmalag
        away_team: Bortalag
        use_context: Om matchkontext ska användas
        
    Returns:
        Tuple med (justerade sannolikheter, kontext-data)
    """
    if not use_context or not HAS_CONTEXT:
        return probs, None
    
    try:
        # Hämta matchkontext med AI
        context = get_match_context(home_team, away_team, use_ai=True)
        
        if context['source'] == 'fallback':
            # Ingen AI-data tillgänglig
            return probs, context
        
        # Beräkna justeringsfaktor
        # Skador minskar chansen att vinna
        home_injury_impact = -context['home_injuries'] / 100.0  # Max -10%
        away_injury_impact = -context['away_injuries'] / 100.0
        
        # Form påverkar också
        form_diff = (context['home_form'] - context['away_form']) / 10.0  # -1 till 1
        form_impact = form_diff * 0.05  # Max ±5%
        
        # Totalt: justera hemmavinst-sannolikheten
        home_adjustment = home_injury_impact + away_injury_impact + form_impact
        home_adjustment += context.get('prediction_adjustment', 0.0)
        
        # Applicera justering
        adjusted_probs = probs.copy()
        adjusted_probs[0] += home_adjustment  # Hemmavinst
        adjusted_probs[2] -= home_adjustment * 0.5  # Bortavinst (hälften av effekten)
        
        # Normalisera så att summan är 1.0
        adjusted_probs = np.maximum(adjusted_probs, 0.01)  # Minimum 1%
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
        
        logger.info(f"Justerade sannolikheter för {home_team} vs {away_team}: {probs} → {adjusted_probs}")
        
        return adjusted_probs, context
        
    except Exception as e:
        logger.error(f"Fel vid justering med kontext: {e}")
        return probs, None


def format_context_summary(context: Optional[Dict]) -> str:
    """
    Formaterar matchkontext till en läsbar sammanfattning
    
    Args:
        context: Kontext-data från get_match_context
        
    Returns:
        Formaterad sträng
    """
    if not context or context['source'] == 'fallback':
        return "Ingen aktuell matchinformation tillgänglig."
    
    summary = []
    
    # Skador
    if context['home_injuries'] > 0 or context['away_injuries'] > 0:
        summary.append(f"**Skador:** Hemma {context['home_injuries']}/10, Borta {context['away_injuries']}/10")
    
    # Form
    summary.append(f"**Form:** Hemma {context['home_form']}/10, Borta {context['away_form']}/10")
    
    # Problem
    if context['home_issues'] != 'Ingen information tillgänglig':
        summary.append(f"**Hemmalag:** {context['home_issues']}")
    if context['away_issues'] != 'Ingen information tillgänglig':
        summary.append(f"**Bortalag:** {context['away_issues']}")
    
    # Justering
    if abs(context.get('prediction_adjustment', 0.0)) > 0.01:
        adj = context['prediction_adjustment']
        if adj > 0:
            summary.append(f"**Bedömning:** Gynnar hemmalaget (+{adj:.1%})")
        else:
            summary.append(f"**Bedömning:** Gynnar bortalaget ({adj:.1%})")
    
    return "\n\n".join(summary)


# Exempel på användning
if __name__ == "__main__":
    # Testa funktionen
    test_probs = np.array([0.45, 0.30, 0.25])
    
    print("=== TEST AV KONTEXTJUSTERING ===\n")
    print(f"Ursprungliga sannolikheter: H={test_probs[0]:.1%}, D={test_probs[1]:.1%}, A={test_probs[2]:.1%}")
    
    adjusted, context = adjust_probabilities_with_context(
        test_probs,
        "Arsenal",
        "Liverpool",
        use_context=True
    )
    
    print(f"Justerade sannolikheter: H={adjusted[0]:.1%}, D={adjusted[1]:.1%}, A={adjusted[2]:.1%}")
    
    if context:
        print("\n" + format_context_summary(context))
