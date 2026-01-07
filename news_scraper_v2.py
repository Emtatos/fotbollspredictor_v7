"""
Intelligent news scraper som använder AI för att analysera nyheter och skador
Istället för att scrapa direkt, använder vi LLM för att sammanfatta aktuell information
"""
import logging
from typing import Dict, Optional
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importera OpenAI om tillgängligt
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI inte installerat. AI-funktioner kommer inte att fungera.")


class IntelligentFootballAnalyzer:
    """
    Intelligent fotbollsanalysator som använder AI för att hämta och analysera
    aktuell information om lag
    """
    
    def __init__(self):
        self.client = None
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            self.client = OpenAI()  # API key och base URL redan konfigurerade
            logger.info("OpenAI-klient initialiserad")
        else:
            logger.warning("OpenAI inte tillgängligt. Använder fallback-metod.")
    
    def get_team_analysis(self, home_team: str, away_team: str) -> Dict:
        """
        Hämtar aktuell analys för en match med AI
        
        Args:
            home_team: Hemmalag
            away_team: Bortalag
            
        Returns:
            Dict med matchanalys
        """
        if not self.client:
            return self._get_fallback_analysis(home_team, away_team)
        
        try:
            logger.info(f"Analyserar match: {home_team} vs {away_team}")
            
            prompt = f"""Analysera följande fotbollsmatch och ge en kort sammanfattning:

Match: {home_team} vs {away_team}

Ge information om:
1. Skador och avstängningar (viktiga spelare som saknas)
2. Senaste formen (senaste 5 matcherna)
3. Tränarbyten eller klubbproblem
4. Andra faktorer som kan påverka matchen

Svara i följande JSON-format:
{{
    "home_injuries": <antal skador 0-10>,
    "away_injuries": <antal skador 0-10>,
    "home_form": <form 0-10>,
    "away_form": <form 0-10>,
    "home_issues": "<kort beskrivning av problem>",
    "away_issues": "<kort beskrivning av problem>",
    "prediction_adjustment": <-0.2 till 0.2, hur mycket detta påverkar oddsen>
}}

Om du inte har aktuell information, gissa baserat på typiska mönster för dessa lag."""
            
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",  # Snabbare och billigare
                messages=[
                    {"role": "system", "content": "Du är en fotbollsexpert som analyserar matcher."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parsa JSON-svar
            import json
            result_text = response.choices[0].message.content
            
            # Extrahera JSON från svaret
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                analysis = json.loads(json_str)
                analysis['timestamp'] = datetime.now().isoformat()
                analysis['source'] = 'AI'
                return analysis
            else:
                logger.warning("Kunde inte parsa JSON från AI-svar")
                return self._get_fallback_analysis(home_team, away_team)
                
        except Exception as e:
            logger.error(f"Fel vid AI-analys: {e}")
            return self._get_fallback_analysis(home_team, away_team)
    
    def _get_fallback_analysis(self, home_team: str, away_team: str) -> Dict:
        """
        Fallback-analys när AI inte är tillgängligt
        Returnerar neutrala värden
        """
        return {
            'home_injuries': 0,
            'away_injuries': 0,
            'home_form': 5,
            'away_form': 5,
            'home_issues': 'Ingen information tillgänglig',
            'away_issues': 'Ingen information tillgänglig',
            'prediction_adjustment': 0.0,
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback'
        }
    
    def get_injury_adjustment(self, home_team: str, away_team: str) -> float:
        """
        Beräknar en justeringsfaktor baserat på skador och nyheter
        
        Args:
            home_team: Hemmalag
            away_team: Bortalag
            
        Returns:
            Justeringsfaktor (-1.0 till 1.0)
            Positivt = gynnar hemmalaget
            Negativt = gynnar bortalaget
        """
        analysis = self.get_team_analysis(home_team, away_team)
        
        # Beräkna justeringsfaktor
        home_impact = (analysis['home_injuries'] / 10.0) * -0.1  # Skador minskar chansen
        away_impact = (analysis['away_injuries'] / 10.0) * 0.1   # Bortalagets skador ökar hemmalagets chans
        
        form_impact = ((analysis['home_form'] - analysis['away_form']) / 10.0) * 0.05
        
        total_adjustment = home_impact + away_impact + form_impact + analysis.get('prediction_adjustment', 0.0)
        
        # Begränsa till -1.0 till 1.0
        return max(-1.0, min(1.0, total_adjustment))


# Enkel cache för att undvika för många API-anrop
_analysis_cache = {}

def get_match_context(home_team: str, away_team: str, use_ai: bool = True) -> Dict:
    """
    Hämtar matchkontext (skador, form, nyheter) för en match
    
    Args:
        home_team: Hemmalag
        away_team: Bortalag
        use_ai: Om AI ska användas (kräver OpenAI API-nyckel)
        
    Returns:
        Dict med matchkontext
    """
    cache_key = f"{home_team}_vs_{away_team}"
    
    # Kontrollera cache (giltig i 1 timme)
    if cache_key in _analysis_cache:
        cached = _analysis_cache[cache_key]
        cache_time = datetime.fromisoformat(cached['timestamp'])
        if (datetime.now() - cache_time).total_seconds() < 3600:
            logger.info(f"Använder cachad analys för {cache_key}")
            return cached
    
    # Hämta ny analys
    analyzer = IntelligentFootballAnalyzer()
    
    if use_ai and analyzer.client:
        analysis = analyzer.get_team_analysis(home_team, away_team)
    else:
        analysis = analyzer._get_fallback_analysis(home_team, away_team)
    
    # Spara i cache
    _analysis_cache[cache_key] = analysis
    
    return analysis


def main():
    """Testfunktion"""
    print("\n=== TEST AV INTELLIGENT FOTBOLLSANALYS ===\n")
    
    analyzer = IntelligentFootballAnalyzer()
    
    # Testa några matcher
    test_matches = [
        ("Arsenal", "Liverpool"),
        ("Manchester City", "Chelsea"),
        ("Newcastle", "Tottenham")
    ]
    
    for home, away in test_matches:
        print(f"\n--- {home} vs {away} ---")
        analysis = analyzer.get_team_analysis(home, away)
        
        print(f"Hemmalag skador: {analysis['home_injuries']}/10")
        print(f"Bortalag skador: {analysis['away_injuries']}/10")
        print(f"Hemmalag form: {analysis['home_form']}/10")
        print(f"Bortalag form: {analysis['away_form']}/10")
        print(f"Hemmalag problem: {analysis['home_issues']}")
        print(f"Bortalag problem: {analysis['away_issues']}")
        print(f"Justeringsfaktor: {analysis.get('prediction_adjustment', 0.0)}")
        
        adjustment = analyzer.get_injury_adjustment(home, away)
        print(f"Total justering: {adjustment:.3f}")
        
        if adjustment > 0:
            print(f"→ Gynnar {home}")
        elif adjustment < 0:
            print(f"→ Gynnar {away}")
        else:
            print("→ Neutral påverkan")


if __name__ == "__main__":
    main()
