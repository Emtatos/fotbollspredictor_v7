"""
Modul för att hämta skador, suspenderingar och annan mänsklig påverkan

Använder API-Football för att hämta:
- Skador
- Suspenderingar
- Tränarbyte
- Spelarbetyg

Körs on-demand via Streamlit-knapp innan tipprundor.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json

from utils import normalize_team_name

logger = logging.getLogger(__name__)


class InjuryDataFetcher:
    """Hämtar och hanterar skadedata från API-Football"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialiserar fetcher med API-nyckel
        
        Args:
            api_key: API-nyckel för API-Football (hämtas från env om None)
        """
        self.api_key = api_key or os.getenv('API_FOOTBALL_KEY')
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {'x-apisports-key': self.api_key} if self.api_key else {}
        
        # Premier League = 39, Championship = 40, League One = 41, League Two = 42
        self.league_ids = {
            'Premier League': 39,
            'Championship': 40,
            'League One': 41,
            'League Two': 42
        }
        
        # Nyckelpositioner (viktigare spelare)
        self.key_positions = ['Goalkeeper', 'Defender', 'Midfielder', 'Attacker']
        
    def fetch_injuries(self, league: str = 'Premier League', season: int = 2024) -> Dict:
        """
        Hämtar skador för en specifik liga
        
        Args:
            league: Ligans namn
            season: Säsong (år)
            
        Returns:
            Dict med skador per lag
        """
        if not self.api_key:
            logger.warning("API_FOOTBALL_KEY saknas - returnerar tom data")
            return {}
        
        league_id = self.league_ids.get(league)
        if not league_id:
            logger.error(f"Okänd liga: {league}")
            return {}
        
        url = f"{self.base_url}/injuries"
        params = {
            'league': league_id,
            'season': season
        }
        
        try:
            logger.info(f"Hämtar skador för {league} (säsong {season})...")
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('errors'):
                logger.error(f"API-fel: {data['errors']}")
                return {}
            
            # Bearbeta skador per lag
            injuries_by_team = {}
            for injury in data.get('response', []):
                team_name = injury['team']['name']
                player_name = injury['player']['name']
                player_type = injury['player'].get('type', 'Unknown')
                reason = injury['player'].get('reason', 'Unknown')
                
                # Normalisera lagnamn
                normalized_team = normalize_team_name(team_name)
                
                if normalized_team not in injuries_by_team:
                    injuries_by_team[normalized_team] = []
                
                injuries_by_team[normalized_team].append({
                    'player': player_name,
                    'position': player_type,
                    'reason': reason,
                    'is_key_player': player_type in self.key_positions
                })
            
            logger.info(f"✅ Hämtade skador för {len(injuries_by_team)} lag")
            return injuries_by_team
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Fel vid API-anrop: {e}")
            return {}
    
    def fetch_all_leagues_injuries(self, season: int = 2024) -> Dict:
        """
        Hämtar skador för alla ligor
        
        Args:
            season: Säsong (år)
            
        Returns:
            Dict med alla skador
        """
        all_injuries = {}
        
        for league in self.league_ids.keys():
            injuries = self.fetch_injuries(league, season)
            all_injuries.update(injuries)
        
        return all_injuries
    
    def calculate_injury_impact(self, team_name: str, injuries_data: Dict) -> Dict:
        """
        Beräknar påverkan av skador för ett lag
        
        Args:
            team_name: Lagets namn
            injuries_data: Skadedata från fetch_injuries
            
        Returns:
            Dict med impact-metrics
        """
        normalized_team = normalize_team_name(team_name)
        team_injuries = injuries_data.get(normalized_team, [])
        
        total_injured = len(team_injuries)
        key_players_injured = sum(1 for inj in team_injuries if inj['is_key_player'])
        
        # Beräkna severity (0-10 skala)
        # Fler skadade nyckelspelare = högre severity
        severity = min(10, key_players_injured * 2 + (total_injured - key_players_injured) * 0.5)
        
        return {
            'total_injured': total_injured,
            'key_players_injured': key_players_injured,
            'injury_severity': round(severity, 2),
            'injured_players': [inj['player'] for inj in team_injuries]
        }
    
    def save_injuries_to_file(self, injuries_data: Dict, filepath: str = 'data/injuries_latest.json'):
        """
        Sparar skadedata till fil
        
        Args:
            injuries_data: Skadedata att spara
            filepath: Sökväg till fil
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Lägg till metadata
        output = {
            'last_updated': datetime.now().isoformat(),
            'injuries': injuries_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Skadedata sparad till {filepath}")
    
    def load_injuries_from_file(self, filepath: str = 'data/injuries_latest.json') -> Optional[Dict]:
        """
        Laddar skadedata från fil
        
        Args:
            filepath: Sökväg till fil
            
        Returns:
            Skadedata eller None om filen inte finns
        """
        if not os.path.exists(filepath):
            logger.warning(f"Skadedatafil saknas: {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            last_updated = data.get('last_updated')
            if last_updated:
                logger.info(f"Skadedata laddad (senast uppdaterad: {last_updated})")
            
            return data.get('injuries', {})
            
        except Exception as e:
            logger.error(f"Fel vid laddning av skadedata: {e}")
            return None
    
    def is_data_stale(self, filepath: str = 'data/injuries_latest.json', hours: int = 24) -> bool:
        """
        Kollar om skadedata är för gammal
        
        Args:
            filepath: Sökväg till fil
            hours: Antal timmar innan data anses gammal
            
        Returns:
            True om data är gammal eller saknas
        """
        if not os.path.exists(filepath):
            return True
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            last_updated_str = data.get('last_updated')
            if not last_updated_str:
                return True
            
            last_updated = datetime.fromisoformat(last_updated_str)
            age = datetime.now() - last_updated
            
            return age > timedelta(hours=hours)
            
        except Exception as e:
            logger.error(f"Fel vid kontroll av data-ålder: {e}")
            return True


def update_injury_data(api_key: Optional[str] = None) -> bool:
    """
    Convenience-funktion för att uppdatera skadedata
    
    Args:
        api_key: API-nyckel (hämtas från env om None)
        
    Returns:
        True om uppdatering lyckades
    """
    try:
        fetcher = InjuryDataFetcher(api_key)
        
        # Hämta skador för alla ligor
        injuries = fetcher.fetch_all_leagues_injuries()
        
        if not injuries:
            logger.warning("Ingen skadedata hämtades")
            return False
        
        # Spara till fil
        fetcher.save_injuries_to_file(injuries)
        
        return True
        
    except Exception as e:
        logger.error(f"Fel vid uppdatering av skadedata: {e}")
        return False


def get_injury_features_for_match(home_team: str, away_team: str, 
                                   injuries_data: Optional[Dict] = None) -> Dict:
    """
    Hämtar skade-features för en specifik match
    
    Args:
        home_team: Hemmalag
        away_team: Bortalag
        injuries_data: Skadedata (laddar från fil om None)
        
    Returns:
        Dict med features för matchen
    """
    fetcher = InjuryDataFetcher()
    
    # Ladda data om inte angiven
    if injuries_data is None:
        injuries_data = fetcher.load_injuries_from_file()
        if injuries_data is None:
            # Returnera default-värden om ingen data finns
            return {
                'InjuredPlayers_Home': 0,
                'InjuredPlayers_Away': 0,
                'KeyPlayersOut_Home': 0,
                'KeyPlayersOut_Away': 0,
                'InjurySeverity_Home': 0,
                'InjurySeverity_Away': 0
            }
    
    # Beräkna impact för båda lagen
    home_impact = fetcher.calculate_injury_impact(home_team, injuries_data)
    away_impact = fetcher.calculate_injury_impact(away_team, injuries_data)
    
    return {
        'InjuredPlayers_Home': home_impact['total_injured'],
        'InjuredPlayers_Away': away_impact['total_injured'],
        'KeyPlayersOut_Home': home_impact['key_players_injured'],
        'KeyPlayersOut_Away': away_impact['key_players_injured'],
        'InjurySeverity_Home': home_impact['injury_severity'],
        'InjurySeverity_Away': away_impact['injury_severity']
    }


if __name__ == "__main__":
    # Test-kod
    logging.basicConfig(level=logging.INFO)
    
    print("Testar injury_scraper...")
    success = update_injury_data()
    
    if success:
        print("\n✅ Skadedata uppdaterad!")
        
        # Testa att ladda data
        fetcher = InjuryDataFetcher()
        injuries = fetcher.load_injuries_from_file()
        
        if injuries:
            print(f"\nAntal lag med skador: {len(injuries)}")
            
            # Visa exempel
            for team, team_injuries in list(injuries.items())[:3]:
                print(f"\n{team}: {len(team_injuries)} skadade")
                for inj in team_injuries[:2]:
                    print(f"  - {inj['player']} ({inj['position']}): {inj['reason']}")
    else:
        print("\n❌ Kunde inte uppdatera skadedata")
