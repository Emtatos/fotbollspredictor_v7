"""
Injury Scraper - Hämtar skador och suspenderingar från API-Football

Detta modul hämtar aktuell skadedata för fotbollslag och beräknar
hur mycket skador påverkar lagets styrka.
"""

import requests
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Liga-ID:n för API-Football
LEAGUE_IDS = {
    'Premier League': 39,
    'Championship': 40,
    'League One': 41,
    'League Two': 42
}

class InjuryDataFetcher:
    """Klass för att hämta och hantera skadedata från API-Football"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialiserar InjuryDataFetcher
        
        Args:
            api_key: API-nyckel för API-Football (hämtas från miljövariabel om None)
        """
        self.api_key = api_key or os.getenv('API_FOOTBALL_KEY')
        self.base_url = 'https://v3.football.api-sports.io'
        self.headers = {
            'x-rapidapi-key': self.api_key if self.api_key else '',
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }
        self.cache_file = Path('data/injuries_latest.json')
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    def fetch_injuries_for_league(self, league_id: int, season: int = 2025) -> List[Dict]:
        """
        Hämtar skador för en specifik liga
        
        Args:
            league_id: ID för ligan i API-Football
            season: Säsong (år)
            
        Returns:
            Lista med skadedata
        """
        if not self.api_key:
            logger.warning("Ingen API-nyckel tillgänglig för API-Football")
            return []
        
        url = f"{self.base_url}/injuries"
        params = {
            'league': league_id,
            'season': season
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('response'):
                logger.info(f"Hämtade {len(data['response'])} skador för liga {league_id}")
                return data['response']
            else:
                logger.warning(f"Ingen skadedata för liga {league_id}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Fel vid hämtning av skadedata: {e}")
            return []
    
    def fetch_all_injuries(self) -> Dict[str, List[Dict]]:
        """
        Hämtar skador för alla engelska ligor
        
        Returns:
            Dictionary med lagnamn som nycklar och skadedata som värden
        """
        all_injuries = {}
        
        for league_name, league_id in LEAGUE_IDS.items():
            logger.info(f"Hämtar skador för {league_name}...")
            injuries = self.fetch_injuries_for_league(league_id)
            
            for injury in injuries:
                team_name = injury['team']['name']
                if team_name not in all_injuries:
                    all_injuries[team_name] = []
                all_injuries[team_name].append(injury)
        
        return all_injuries
    
    def save_to_cache(self, data: Dict) -> None:
        """Sparar skadedata till cache-fil"""
        cache_data = {
            'last_updated': datetime.now().isoformat(),
            'data': data
        }
        
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Skadedata sparad till {self.cache_file}")
    
    def load_from_cache(self) -> Optional[Dict]:
        """Laddar skadedata från cache-fil"""
        if not self.cache_file.exists():
            return None
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Kunde inte ladda cache: {e}")
            return None
    
    def is_data_stale(self, max_age_hours: int = 24) -> bool:
        """
        Kontrollerar om cachad data är för gammal
        
        Args:
            max_age_hours: Maximal ålder i timmar
            
        Returns:
            True om data är för gammal eller saknas
        """
        cache = self.load_from_cache()
        if not cache:
            return True
        
        last_updated = datetime.fromisoformat(cache['last_updated'])
        age = datetime.now() - last_updated
        
        return age > timedelta(hours=max_age_hours)
    
    def calculate_injury_impact(self, team_name: str, injuries_data: Dict) -> Dict:
        """
        Beräknar hur mycket skador påverkar ett lag
        
        Args:
            team_name: Lagnamn
            injuries_data: Skadedata för alla lag
            
        Returns:
            Dictionary med injury impact metrics
        """
        team_injuries = injuries_data.get(team_name, [])
        
        if not team_injuries:
            return {
                'total_injured': 0,
                'key_players_injured': 0,
                'injury_severity': 0.0,
                'injured_players': []
            }
        
        # Räkna skadade spelare
        injured_players = []
        key_players_count = 0
        
        for injury in team_injuries:
            player_name = injury['player']['name']
            reason = injury['player']['reason']
            
            # Anta att spelare som är "Out" eller har längre skador är nyckelspelare
            # Detta är en förenkling - i verkligheten skulle vi behöva spelarbetyg
            is_key_player = 'Out' in reason or 'Doubtful' in reason
            
            injured_players.append({
                'name': player_name,
                'reason': reason,
                'is_key': is_key_player
            })
            
            if is_key_player:
                key_players_count += 1
        
        # Beräkna severity (0-10 skala)
        # Nyckelspelare väger tyngre
        severity = min(10, key_players_count * 2 + (len(injured_players) - key_players_count) * 0.5)
        
        return {
            'total_injured': len(injured_players),
            'key_players_injured': key_players_count,
            'injury_severity': round(severity, 1),
            'injured_players': injured_players
        }


def update_injury_data() -> bool:
    """
    Convenience-funktion för att uppdatera skadedata
    
    Returns:
        True om uppdatering lyckades
    """
    fetcher = InjuryDataFetcher()
    
    if not fetcher.api_key:
        logger.warning("Ingen API-nyckel - kan inte uppdatera skadedata")
        return False
    
    try:
        injuries = fetcher.fetch_all_injuries()
        fetcher.save_to_cache(injuries)
        return True
    except Exception as e:
        logger.error(f"Fel vid uppdatering av skadedata: {e}")
        return False


def get_injury_features_for_match(home_team: str, away_team: str) -> Dict:
    """
    Hämtar injury features för en specifik match
    
    Args:
        home_team: Hemmalag
        away_team: Bortalag
        
    Returns:
        Dictionary med injury features för båda lagen
    """
    fetcher = InjuryDataFetcher()
    cache = fetcher.load_from_cache()
    
    if not cache:
        logger.warning("Ingen cachad skadedata - returnerar nollvärden")
        return {
            'InjuredPlayers_Home': 0,
            'InjuredPlayers_Away': 0,
            'KeyPlayersOut_Home': 0,
            'KeyPlayersOut_Away': 0,
            'InjurySeverity_Home': 0.0,
            'InjurySeverity_Away': 0.0
        }
    
    injuries_data = cache.get('data', {})
    
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


if __name__ == '__main__':
    # Test
    logging.basicConfig(level=logging.INFO)
    
    print("Testar InjuryDataFetcher...")
    fetcher = InjuryDataFetcher()
    
    if fetcher.api_key:
        print("API-nyckel hittad!")
        print("Hämtar skadedata...")
        success = update_injury_data()
        
        if success:
            print("✅ Skadedata uppdaterad!")
            
            # Testa att hämta features
            features = get_injury_features_for_match('Arsenal', 'Chelsea')
            print(f"\nSkade-features för Arsenal vs Chelsea:")
            for key, value in features.items():
                print(f"  {key}: {value}")
        else:
            print("❌ Kunde inte uppdatera skadedata")
    else:
        print("⚠️ Ingen API-nyckel - sätt API_FOOTBALL_KEY i miljövariabler")
